"""
RAG Indexing Pipeline
─────────────────────
download_and_process(document) runs all stages in sequence:

  1. download_pdf   — fetch PDF bytes from URL, save to disk
  2. extract_text   — pull text out of the PDF page by page
  3. chunk_text     — split text into overlapping windows
  4. embed_and_store — generate vectors, upsert into Pinecone
"""


import os
import re
import uuid
import requests
import pymupdf
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from django.conf import settings


# Embedding model (loaded once at import time) 
_embedder = SentenceTransformer('all-MiniLM-L6-v2')  

# PUBLIC ENTRY POINT
def process_document(document):
    "Runs document pipeline, updates status, and saves errors_msg on failure."

    try:
        document.status = 'processing'
        document.save(update_fields=['status'])

        file_path  = _download_pdf(document.url, document.id)
        pages      = _extract_text(file_path)
        chunks     = _chunk_pages(pages)
        namespace  = _embed_and_store(document.id, chunks)

        # Persist results
        from .models import DocumentChunk
        DocumentChunk.objects.bulk_create([
            DocumentChunk(
                document=document,
                text=c['text'],
                chunk_index=i,
                page_number=c['page'],
            )
            for i, c in enumerate(chunks)
        ])

        document.file_path          = file_path
        document.page_count         = len(pages)
        document.chunk_count        = len(chunks)
        document.pinecone_namespace = namespace
        document.status             = 'ready'
        document.save()

    except Exception as exc:
        document.status        = 'failed'
        document.error_message = str(exc)
        document.save(update_fields=['status', 'error_message'])
        raise
    

def _download_pdf(url: str, doc_id: int) -> str:
    """Downloads a PDF via streaming so large files are never fully loaded into RAM, validates its %PDF header, and saves it to media/pdfs/, returning the absolute file path."""    
    
    save_dir = os.path.join(settings.MEDIA_ROOT, 'pdfs')
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"doc_{doc_id}_{uuid.uuid4().hex[:6]}.pdf")

    response = requests.get(url, stream=True, timeout=60,
                            headers={'User-Agent': 'RAGBot/1.0'})
    response.raise_for_status()

    with open(path, 'wb') as f:
        for chunk in response.iter_content(8192):
            f.write(chunk)

    # Reject anything that isn't really a PDF
    with open(path, 'rb') as f:
        if f.read(4) != b'%PDF':
            os.remove(path)
            raise ValueError("URL did not return a valid PDF file.")

    return path     