from rest_framework import serializers
from .models import Conversation, Message


class MessageSerializer(serializers.ModelSerializer):
    class Meta:
        model  = Message
        fields = ['id', 'role', 'content', 'source_chunks', 'created_at']


class ConversationSerializer(serializers.ModelSerializer):
    messages = MessageSerializer(many=True, read_only=True)

    class Meta:
        model  = Conversation
        fields = ['id', 'document', 'title', 'messages', 'created_at', 'updated_at']


class AskSerializer(serializers.Serializer):
    question        = serializers.CharField(min_length=2, max_length=2000)
    document_id     = serializers.IntegerField()
    conversation_id = serializers.IntegerField(required=False, allow_null=True)


# class AnswerSerializer(serializers.Serializer):
#     answer          = serializers.CharField()
#     conversation_id = serializers.IntegerField()
#     source_chunks   = serializers.ListField()