# print(__doc__)
"""
RAG Query Pipeline

answer_question(question, document, history) → str

  1. embed_query    — convert the question to a vector
  2. search_pinecone — find the top-k most similar chunks
  3. fetch_chunks   — load full text from Django DB
  4. build_prompt   — assemble context + history + question
  5. call_llm       — stream response from Groq
"""


from groq import Groq
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from django.conf import settings
from apps.documents.models import DocumentChunk


# --- PERSISTENT CLIENT POOLING ---
# Instantiating these at the module level reuses underlying connection pools,
# avoiding costly cryptographic handshakes on every single query.
# _embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Thread-safe global variable for lazy loading
_embedding_model = None

def get_embedding_model() -> SentenceTransformer:
    """Lazy loads the embedding model to optimize startup time and memory footprint."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _embedding_model


_pinecone_client = Pinecone(api_key=settings.PINECONE_API_KEY)
_vector_index = _pinecone_client.Index(settings.PINECONE_INDEX_NAME)

_groq_client = Groq(api_key=settings.GROQ_API_KEY)


# PUBLIC ENTRY POINT
def answer_question(question: str, document, history: list[dict]) -> dict:
    """
    Full query pipeline modified for streaming. Returns:
        {
          'answer_stream':  generator (yields str chunks),
          'source_chunks': [ {chunk_index, page_number, score, preview} ]
        }
    """

    query_vec = _embed_query(question)
    matches   = _search_pinecone(query_vec, document.pinecone_namespace)

    if not matches:       # to stream fallback response
        def _fallback_stream():
            yield "I couldn't find relevant information in this document for your question."
        
        return {
            'answer_stream': _fallback_stream(),
            'source_chunks': [],
        }
    
    chunks, valid_matches = _fetch_chunks(document.id, matches)

    if not chunks:
        def _fallback_stream():
            yield "I couldn't find relevant information in this document for your question."
        return {
            'answer_stream': _fallback_stream(),
            'source_chunks': [],
        }

    system, user = _build_prompt(question, chunks, history)
    answer_stream       = _call_llm(system, user)

    source_chunks = [
        {
            'chunk_index': m.metadata['chunk_index'],
            'page_number': m.metadata['page'],
            'score':       round(m.score, 4),
            'preview':     m.metadata['preview'],
        }
        for m in valid_matches
    ]

    return {'answer_stream': answer_stream, 'source_chunks': source_chunks}

# 1 — EMBED QUERY
def _embed_query(question: str) -> list[float]:
    """
    Converts the question into a vector using the same model used at index time.
    Using a different model here than during indexing would produce garbage results.
    """
    model = get_embedding_model()
    return model.encode(question, normalize_embeddings=True).tolist()


# 2 — VECTOR SEARCH
def _search_pinecone(query_vec: list[float], namespace: str, top_k: int = 5):
    """
    Queries Pinecone for the top_k vectors most similar to query_vec.
    Restricts search to the document's own namespace so results stay relevant.
    Returns Pinecone match objects (each has .score and .metadata).
    """
    result = _vector_index.query(
        vector=query_vec,
        top_k=top_k,
        namespace=namespace,
        include_metadata=True,
    )
    # Filter out low-confidence matches (below 0.3 cosine similarity)
    return [m for m in result.matches if m.score >= 0.3]


# 3 — FETCH CHUNK TEXT
def _fetch_chunks(doc_id: int, matches) -> tuple[list[str], list]:
    """
    Looks up full text blocks from Django DB using chunk_index values.
    Returns lists of valid string contents and corresponding vector match objects.
    """
    indices = [m.metadata['chunk_index'] for m in matches]
    chunks  = DocumentChunk.objects.filter(document_id=doc_id, chunk_index__in=indices)
    index_to_text = {c.chunk_index: c.text for c in chunks}    # { 5: "AI is...", 8: "Machine learning...", ...}
    
    valid_texts = []
    valid_matches = []
    
    for m in matches:
        text = index_to_text.get(m.metadata['chunk_index'], '').strip()
        if text:  # Ensure text exists to guarantee 1:1 citation alignments
            valid_texts.append(text)
            valid_matches.append(m)
            
    return valid_texts, valid_matches


# 4 — BUILD PROMPT
def _build_prompt(question: str, chunks: list[str],
                  history: list[dict]) -> tuple[str, str]:
    """
    Assembles the final prompt sent to the LLM.

    System prompt: constrains the model to only use provided context.
    User message:  context blocks + optional prior conversation + question.
    """
    system = (
        "You are a helpful assistant that answers questions about a document.\n"
        "Rules:\n"
        "1. Answer using ONLY the context excerpts provided below.\n"
        "2. If the context does not contain enough information, say so clearly.\n"
        "3. Cite the source number (e.g. [Source 2]) when you use a specific excerpt.\n"
        "4. Be concise."
    )

    # Clean and simple alignment guaranteed by step 3
    context_block = "\n\n".join(f"[Source {i+1}]\n{text}" for i, text in enumerate(chunks))

    history_block = ""
    if history:
        lines = [
            f"{'User' if m['role']=='user' else 'Assistant'}: {m['content']}"
            for m in history[-6:]
        ]
        history_block = "\nPrevious conversation:\n" + "\n".join(lines) + "\n"

    user_msg = f"Context:\n{context_block}\n{history_block}\nQuestion: {question}"

    return system, user_msg


# 5 — CALL LLM (GROQ STREAMING)
def _call_llm(system: str, user_msg: str):
    """
    Sends the prompt payload to Groq and yields text chunks as they arrive.
    Uses GROK_API_KEY from environment or settings file.
    """    
    
    # Using llama-3.3-70b-versatile as a highly efficient default for RAG context
    response_stream = _groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg}
        ],
        stream=True,
    )
    
    for chunk in response_stream:
        # Use structural protection against broken chunks or blank finish_reasons
        if chunk.choices and chunk.choices[0].delta:
            # Extract the content delta text safely
            content = chunk.choices[0].delta.content
            if content:
                yield content