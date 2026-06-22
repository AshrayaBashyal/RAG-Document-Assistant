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


