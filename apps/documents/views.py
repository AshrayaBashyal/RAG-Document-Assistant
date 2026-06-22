from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from drf_spectacular.utils import extend_schema, OpenApiParameter

from .models import Document, DocumentChunk
from .serializers import DocumentSerializer, SubmitSerializer, ChunkSerializer
from .services import process_document


class DocumentListView(APIView):
    """List all documents belonging to the current user."""
    permission_classes = [IsAuthenticated]

    @extend_schema(
        responses={200: DocumentSerializer(many=True)},
        summary="List my documents",
        tags=["Documents"],
    )
    def get(self, request):
        docs = Document.objects.filter(user=request.user)
        return Response(DocumentSerializer(docs, many=True).data)


class DocumentSubmitView(APIView):
    """
    Submit a PDF URL to kick off the RAG indexing pipeline.

    The pipeline runs synchronously (blocking).
    In production should push it to a background task queue (Celery/RQ) 
    """
    permission_classes = [IsAuthenticated]

    @extend_schema(
        request=SubmitSerializer,
        responses={201: DocumentSerializer},
        summary="Submit a PDF URL",
        description=(
            "Downloads the PDF, extracts text, chunks it, generates embeddings, "
            "and stores vectors in Pinecone. Returns when the document is ready."
        ),
        tags=["Documents"],
    )
    def post(self, request):
        serializer = SubmitSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        document = Document.objects.create(
            user=request.user,
            url=serializer.validated_data['url'],
            title=serializer.validated_data.get('title') or serializer.validated_data['url'],
        )

        try:
            process_document(document)   # runs the full pipeline
        except Exception as exc:
            # process_document already set status=failed on the document;
            # we just surface the message to the caller here.
            return Response(
                {'error': str(exc), 'document_id': document.id},
                status=status.HTTP_400_BAD_REQUEST,
            )

        return Response(DocumentSerializer(document).data, status=status.HTTP_201_CREATED)


class DocumentDetailView(APIView):
    """Retrieve status and metadata for a single document."""
    permission_classes = [IsAuthenticated]

    @extend_schema(
        responses={200: DocumentSerializer},
        summary="Document detail",
        tags=["Documents"],
    )
    def get(self, request, pk):
        document = self._get_document(request.user, pk)
        if document is None:
            return Response({'error': 'Not found'}, status=status.HTTP_404_NOT_FOUND)
        return Response(DocumentSerializer(document).data)

    @extend_schema(
        responses={204: None},
        summary="Delete a document",
        tags=["Documents"],
    )
    def delete(self, request, pk):
        document = self._get_document(request.user, pk)
        if document is None:
            return Response({'error': 'Not found'}, status=status.HTTP_404_NOT_FOUND)
        document.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)

    @staticmethod
    def _get_document(user, pk):
        try:
            return Document.objects.get(pk=pk, user=user)
        except Document.DoesNotExist:
            return None


class DocumentChunksView(APIView):
    """Return all text chunks for a document (useful for debugging retrieval)."""
    permission_classes = [IsAuthenticated]

    @extend_schema(
        responses={200: ChunkSerializer(many=True)},
        summary="List document chunks",
        tags=["Documents"],
    )
    def get(self, request, pk):
        try:
            document = Document.objects.get(pk=pk, user=request.user)
        except Document.DoesNotExist:
            return Response({'error': 'Not found'}, status=status.HTTP_404_NOT_FOUND)

        chunks = DocumentChunk.objects.filter(document=document)
        return Response(ChunkSerializer(chunks, many=True).data)