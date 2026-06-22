from django.db import models
from django.contrib.auth.models import User
from apps.documents.models import Document


class Conversation(models.Model):
    user       = models.ForeignKey(User, on_delete=models.CASCADE, related_name='conversations')
    document   = models.ForeignKey(Document, on_delete=models.CASCADE, related_name='conversations')
    title      = models.CharField(max_length=200, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-updated_at']


class Message(models.Model):
    class Role(models.TextChoices):
        USER      = 'user',      'User'
        ASSISTANT = 'assistant', 'Assistant'

    conversation = models.ForeignKey(Conversation, on_delete=models.CASCADE,
                                     related_name='messages')
    role         = models.CharField(max_length=10, choices=Role.choices)
    content      = models.TextField()
    # Which chunks were used to produce this answer (assistant messages only)
    source_chunks = models.JSONField(default=list)
    created_at   = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['created_at']