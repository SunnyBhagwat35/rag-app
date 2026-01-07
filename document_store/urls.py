from django.urls import path
from .views import (
    FileUploadView,
    FileListView,
    FileDetailView,
    # BulkFileUploadView,
    # FileStatsView
)

urlpatterns = [
    # Main endpoints
    path('files/', FileListView.as_view(), name='file-list'),
    path('files/upload/', FileUploadView.as_view(), name='file-upload'),
    path('files/<int:pk>/', FileDetailView.as_view(), name='file-detail'),
    
    # # Additional endpoints
    # path('files/bulk-upload/', BulkFileUploadView.as_view(), name='bulk-upload'),
    # path('files/stats/', FileStatsView.as_view(), name='file-stats'),
]