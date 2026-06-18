from django.contrib.auth import get_user_model
from django.db import models

User = get_user_model()


class Document(models.Model):
    """Represents a user-submitted PDF and tracks its processing pipeline state."""

    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        PROCESSING = "processing", "Processing"
        READY = "ready", "Ready"
        FAILED = "failed", "Failed"

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="documents")
    title = models.CharField(max_length=500, blank=True)
    url = models.URLField(max_length=2000)
    file_path = models.CharField(max_length=1000, blank=True)  # Local storage path
    status = models.CharField(
        max_length=20, choices=Status.choices, default=Status.PENDING
    )
    error_message = models.TextField(blank=True)  # System/pipeline error logs

    # Extracted metadata
    page_count = models.IntegerField(default=0)
    chunk_count = models.IntegerField(default=0)
    raw_text = models.TextField(blank=True)

    # Vector isolation (use metadata filter --> metadata: {document_id: X, user_id: Y} for multi-doc searches)
    pinecone_namespace = models.CharField(max_length=200, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.title or self.url} ({self.status})"


class DocumentChunk(models.Model):
    """Stores text chunks locally to retrieve content after Pinecone vector queries."""

    document = models.ForeignKey(
        Document, on_delete=models.CASCADE, related_name="chunks"
    )
    text = models.TextField()

    # Position metadata
    chunk_index = models.IntegerField()  # 0-indexed sequential order
    page_number = models.IntegerField(default=0)

    # After embedding, we store the Pinecone vector ID so we can cross-reference
    pinecone_vector_id = models.CharField(max_length=200, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["chunk_index"]

    def __str__(self):
        return f"Chunk {self.chunk_index} of {self.document.title}"