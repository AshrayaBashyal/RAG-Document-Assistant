import json
from django.http import StreamingHttpResponse
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from drf_spectacular.utils import extend_schema

from apps.documents.models import Document
from .models import Conversation, Message
from .serializers import AskSerializer, ConversationSerializer
from .services import answer_question

class AskView(APIView):
    """
    Ask a question about a document and stream the response chunk-by-chunk.
    
    Outputs a line-delimited JSON stream.
    """
    permission_classes = [IsAuthenticated]

    @extend_schema(
        request=AskSerializer,
        responses={200: "Line-delimited JSON stream of response chunks"},
        summary="Ask a question (Streaming)",
        tags=["Q&A"],
    )
    def post(self, request):
        serializer = AskSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data

        # Validate document 
        try:
            document = Document.objects.get(pk=data['document_id'], user=request.user)
        except Document.DoesNotExist:
            return StreamingHttpResponse(
                [json.dumps({'error': 'Document not found'})], 
                status=status.HTTP_404_NOT_FOUND, 
                content_type="application/json"
            )

        if document.status != 'ready':
            return StreamingHttpResponse(
                [json.dumps({'error': f"Document is not ready (status: {document.status})"})], 
                status=status.HTTP_400_BAD_REQUEST, 
                content_type="application/json"
            )

        # Get or create conversation 
        conv_id = data.get('conversation_id')
        if conv_id:
            try:
                conversation = Conversation.objects.get(pk=conv_id, user=request.user)
            except Conversation.DoesNotExist:
                return StreamingHttpResponse(
                    [json.dumps({'error': 'Conversation not found'})], 
                    status=status.HTTP_404_NOT_FOUND,  
                    content_type="application/json"
                )
        else:
            conversation = Conversation.objects.create(
                user=request.user,
                document=document,
                title=data['question'][:80],
            )

        # Save user message 
        Message.objects.create(
            conversation=conversation,
            role=Message.Role.USER,
            content=data['question'],
        )

        # Build history for context 
        history = list(
            conversation.messages
            .values('role', 'content')
            .order_by('created_at')
        )

        # Run RAG pipeline (Returns generators) 
        result = answer_question(data['question'], document, history)
        
        answer_stream = result['answer_stream']
        source_chunks = result['source_chunks']

        # Streaming Event Generator 
        def stream_response():
            # 1. Immediately yield the source references so the UI can render them
            yield json.dumps({
                "type": "metadata",
                "conversation_id": conversation.id,
                "source_chunks": source_chunks
            }) + "\n"

            # 2. Iterate through the live LLM tokens and compile the final text
            full_answer = ""
            for chunk in answer_stream:
                full_answer += chunk
                yield json.dumps({"type": "chunk", "text": chunk}) + "\n"

            # 3. Stream completed: Persist the assistant's complete response to DB
            if full_answer.strip():
                Message.objects.create(
                    conversation=conversation,
                    role=Message.Role.ASSISTANT,
                    content=full_answer,
                    source_chunks=source_chunks,
                )

        # Return the response stream with streaming headers
        response = StreamingHttpResponse(stream_response(), content_type="application/x-ndjson")
        response['X-Accel-Buffering'] = 'no'  # Prevents Nginx from caching chunks
        return response
    

   