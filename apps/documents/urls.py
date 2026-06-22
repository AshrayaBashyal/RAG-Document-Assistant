from django.urls import path
from .views import DocumentListView, DocumentSubmitView, DocumentDetailView, DocumentChunksView

urlpatterns = [
    path('',              DocumentListView.as_view(),   name='document-list'),
    path('submit/',       DocumentSubmitView.as_view(),  name='document-submit'),
    path('<int:pk>/',     DocumentDetailView.as_view(),  name='document-detail'),
    path('<int:pk>/chunks/', DocumentChunksView.as_view(), name='document-chunks'),
]