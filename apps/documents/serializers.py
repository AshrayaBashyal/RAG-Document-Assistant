from rest_framework import serializers
from .models import Document, DocumentChunk


class DocumentSerializer(serializers.ModelSerializer):
    class Meta:
        model  = Document
        fields = ['id', 'title', 'url', 'status', 'error_message',
                  'page_count', 'chunk_count', 'created_at']
        read_only_fields = ['status', 'error_message', 'page_count',
                            'chunk_count', 'created_at']


class SubmitSerializer(serializers.Serializer):
    url   = serializers.URLField()
    title = serializers.CharField(max_length=200, required=False, default='')


class ChunkSerializer(serializers.ModelSerializer):
    class Meta:
        model  = DocumentChunk
        fields = ['id', 'chunk_index', 'page_number', 'text']