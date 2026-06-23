"""
RAG Query Pipeline

answer_question(question, document, history) → str

  1. embed_query    — convert the question to a vector
  2. search_pinecone — find the top-k most similar chunks
  3. fetch_chunks   — load full text from Django DB
  4. build_prompt   — assemble context + history + question
  5. call_llm       — send to Claude and return the answer
"""

# import anthropic/ grok/ gemini ???
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from django.conf import settings
from documents.models import DocumentChunk


_embedder = SentenceTransformer('all-MiniLM-L6-v2')

# PUBLIC ENTRY POINT
def answer_question(question: str, document, history: list[dict]) -> dict:
    """
    Full query pipeline. Returns:
        {
          'answer':       str,
          'source_chunks': [ {chunk_index, page_number, score, preview} ]
        }
    """

    query_vec = _embed_query(question)
    matches   = _search_pinecone(query_vec, document.pinecone_namespace)

    if not matches:
        return {
            'answer': "I couldn't find relevant information in this document for your question.",
            'source_chunks': [],
        }
    
    chunks       = _fetch_chunks(document.id, matches)
    system, user = _build_prompt(question, chunks, history)
    answer       = _call_llm(system, user)

    source_chunks = [
        {
            'chunk_index': m.metadata['chunk_index'],
            'page_number': m.metadata['page'],
            'score':       round(m.score, 4),
            'preview':     m.metadata['preview'],
        }
        for m in matches
    ]

    return {'answer': answer, 'source_chunks': source_chunks}


# 1 — EMBED QUERY
def _embed_query(question: str) -> list[float]:
    """
    Converts the question into a vector using the same model used at index time.
    Using a different model here than during indexing would produce garbage results.
    """
    return _embedder.encode(question, normalize_embeddings=True).tolist()


# 2 — VECTOR SEARCH
def _search_pinecone(query_vec: list[float], namespace: str, top_k: int = 5):
    """
    Queries Pinecone for the top_k vectors most similar to query_vec.
    Restricts search to the document's own namespace so results stay relevant.
    Returns Pinecone match objects (each has .score and .metadata).
    """
    index = Pinecone(api_key=settings.PINECONE_API_KEY).Index(settings.PINECONE_INDEX_NAME)
    result = index.query(
        vector=query_vec,
        top_k=top_k,
        namespace=namespace,
        include_metadata=True,
    )
    # Filter out low-confidence matches (below 0.3 cosine similarity)
    return [m for m in result.matches if m.score >= 0.3]


# 3 — FETCH CHUNK TEXT
def _fetch_chunks(doc_id: int, matches) -> list[str]:
    """
    Pinecone only stores metadata + vectors, not the full text.
    We use chunk_index from the metadata to look up the full text in Django's DB.
    """
    indices = [m.metadata['chunk_index'] for m in matches]
    chunks  = DocumentChunk.objects.filter(document_id=doc_id, chunk_index__in=indices)
    index_to_text = {c.chunk_index: c.text for c in chunks}
    # Return in the same order as matches (highest score first)
    return [index_to_text.get(m.metadata['chunk_index'], '') for m in matches]

