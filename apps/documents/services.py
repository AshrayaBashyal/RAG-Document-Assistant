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

# To only load it when actually generating embeddings. This avoids:slow migrations, slow startup, model loading during tests, unnecessary memory usage

# _embedding_model = None

# def get_embedding_model():
#     global _embedding_model

#     if _embedding_model is None:
#         _embedding_model = SentenceTransformer(
#             "all-MiniLM-L6-v2"
#         )

#     return _embedding_model
# then -----> embeddings = model.encode(texts) <-----

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
    

# DOWNLOAD
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


# EXTRACT
def _extract_text(file_path: str) -> list[dict]:
    """
    Extracts and cleans plain text page-by-page using PyMuPDF.
    Returns a list of {'page': int, 'text': str} dicts.
    Cleans text by collapsing whitespace, stripping stray page numbers,
    and fixing hyphen line-breaks (e.g., "implemen-\ntion" -> "implementation").
    """
 
    doc   = pymupdf.open(file_path)
    pages = []

    for i, page in enumerate(doc):
        raw = page.get_text("text")
        text = _clean(raw)
        if text.strip():
            pages.append({'page': i + 1, 'text': text})

    doc.close()

    if not pages:
        raise ValueError("Could not extract any text. The PDF may be scanned/image-based.")

    return pages


def _clean(text: str) -> str:         # OTHER ADDITIONAL METHODS MAY ALSO NEED TO BE APPLIED
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)   # fix hyphen breaks
    # Removed to prevent NUMERICAL DATA loss in numeric/finance docs
    # text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)  # lone page numbers 
    text = re.sub(r'[ \t]+', ' ', text)               # collapse spaces
    text = re.sub(r'\n{3,}', '\n\n', text)            # max 2 blank lines
    return text.strip()


# CHUNKING
def _chunk_pages(pages: list[dict],
                 chunk_size: int = 1000,
                 overlap: int = 200) -> list[dict]:
    """
    Splits each page's text into overlapping fixed-size character windows.

    chunk_size = how many characters per chunk  (~150-200 words)
    overlap    = how many characters re-used between consecutive chunks
                 (prevents context being lost at a boundary)

    Returns list of {'page': int, 'text': str}.
    """

    chunks = []

    for page in pages:
        text  = page['text']
        start = 0

        while start < len(text):
                                                                        
            end        = min(start + chunk_size, len(text))       
            chunk_text = text[start:end].strip()     # To PREVENT WORDS FRM BEING CUT IN HALF, search backward from the end of your 1000-character window to find the closest blank space.

            if chunk_text:
                chunks.append({'page': page['page'], 'text': chunk_text})

            if end == len(text):
                break
            start = end - overlap   # step back by overlap for next window

    return chunks


# EMBED + STORE
def _embed_and_store(doc_id: int, chunks: list[dict]) -> str:
    """
    Generates an embedding vector for every chunk and upserts them into Pinecone.

    Each document gets its own Pinecone *namespace* so searches stay isolated.
    The vector id encodes both document id and chunk position for easy debugging.

    Returns the namespace string (stored on the Document record).
    """
    namespace = f"doc_{doc_id}_{uuid.uuid4().hex[:6]}"
    texts     = [c['text'] for c in chunks]

    # Batch encode — much faster than one at a time
    vectors = _embedder.encode(texts, normalize_embeddings=True, batch_size=32)

    index = Pinecone(api_key=settings.PINECONE_API_KEY).Index(settings.PINECONE_INDEX_NAME)

    # Pinecone upsert expects list of (id, vector, metadata)
    to_upsert = [
        {
            'id':       f"doc_{doc_id}_chunk_{i}",
            'values':   vec.tolist(),
            'metadata': {
                'doc_id':      doc_id,
                'chunk_index': i,
                'page':        chunks[i]['page'],
                'preview':     chunks[i]['text'][:200],
            }
        }
        for i, vec in enumerate(vectors)
    ]

    # Upsert in batches of 100 (Pinecone recommended limit per request)
    for i in range(0, len(to_upsert), 100):
        index.upsert(vectors=to_upsert[i:i+100], namespace=namespace)

    return namespace