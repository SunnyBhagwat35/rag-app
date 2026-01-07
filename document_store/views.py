from rest_framework.views import APIView
from rest_framework.status import HTTP_201_CREATED, HTTP_200_OK, HTTP_400_BAD_REQUEST, HTTP_204_NO_CONTENT
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.views import APIView
from django.db.models import Q
from .models import Document
from .serializers import FileUploadSerializer, FileListSerializer


class FileUploadView(APIView):
    def post(self, request, *args, **kwargs):    
        serializer = FileUploadSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(
                {
                    'message': 'File uploaded successfully',
                    'data': serializer.data
                },
                status=HTTP_201_CREATED
            )
        return Response(
            {
                'message': 'File upload failed',
                'errors': serializer.errors
            },
            status=HTTP_400_BAD_REQUEST
        )


class FileListView(APIView):
    def get(self, request):
        queryset = Document.objects.all()
            
        search = self.request.query_params.get('search', None)
        if search:
            queryset = queryset.filter(
                Q(original_name__icontains=search) |
                Q(description__icontains=search)
            )
        
        file_type = self.request.query_params.get('file_type', None)
        if file_type:
            if not file_type.startswith('.'):
                file_type = f'.{file_type}'
            queryset = queryset.filter(file_type__iexact=file_type)
        
        ordering = self.request.query_params.get('ordering', 'uploaded_at')
        valid_orderings = ['uploaded_at', '-uploaded_at', 'file_size', '-file_size', 
                          'original_name', '-original_name']
        if ordering in valid_orderings:
            queryset = queryset.order_by(ordering)
        
        serializer = FileListSerializer(queryset, many=True)
        
        return Response(
            serializer.data,
            status=HTTP_200_OK
        )


class FileDetailView(APIView):
    def get(self, request, pk, *args, **kwargs):
        instance = Document.objects.get(pk=pk)

        file = FileListSerializer(instance).data
        return Response(
            file,
            status=HTTP_204_NO_CONTENT
        )

    def delete(self, request, pk, *args, **kwargs):  
        instance = Document.objects.get(pk=pk)
        # Delete the actual file from storage
        if instance.file:
            instance.file.delete(save=False)
        # self.perform_destroy(instance)
        return Response(
            {'message': 'File deleted successfully'},
            status=HTTP_204_NO_CONTENT
        )


# class BulkFileUploadView(APIView):
#     """
#     API endpoint for uploading multiple files at once
    
#     POST /api/files/bulk-upload/
#     """    
#     def post(self, request, *args, **kwargs):
#         files = request.FILES.getlist('files')
#         description = request.data.get('description', '')
        
#         if not files:
#             return Response(
#                 {'message': 'No files provided'},
#                 status=HTTP_400_BAD_REQUEST
#             )
        
#         uploaded_files = []
#         errors = []
        
#         for file in files:
#             data = {
#                 'file': file,
#                 'description': description
#             }
#             serializer = FileUploadSerializer(data=data, context={'request': request})
            
#             if serializer.is_valid():
#                 serializer.save()
#                 uploaded_files.append(serializer.data)
#             else:
#                 errors.append({
#                     'file': file.name,
#                     'errors': serializer.errors
#                 })
        
#         response_data = {
#             'uploaded_count': len(uploaded_files),
#             'failed_count': len(errors),
#             'uploaded_files': uploaded_files
#         }
        
#         if errors:
#             response_data['failed_files'] = errors
        
#         status_code = status.HTTP_201_CREATED if uploaded_files else status.HTTP_400_BAD_REQUEST
        
#         return Response(response_data, status=status_code)


# class FileStatsView(APIView):
#     """
#     API endpoint for getting statistics about uploaded files
    
#     GET /api/files/stats/
#     """
#     permission_classes = []  # Allow any user
    
#     def get(self, request):
#         total_files = Document.objects.count()
#         total_size = sum(file.file_size for file in Document.objects.all())
        
#         # Get file type distribution
#         file_types = {}
#         for file in Document.objects.all():
#             ext = file.file_type or 'unknown'
#             file_types[ext] = file_types.get(ext, 0) + 1
        
#         stats = {
#             'total_files': total_files,
#             'total_size_bytes': total_size,
#             'total_size_display': self.format_bytes(total_size),
#             'file_types_distribution': file_types,
#             'latest_upload': Document.objects.first().uploaded_at if total_files > 0 else None
#         }
        
#         return Response(stats)
    
#     def format_bytes(self, size):
#         for unit in ['B', 'KB', 'MB', 'GB']:
#             if size < 1024.0:
#                 return f"{size:.2f} {unit}"
#             size /= 1024.0
#         return f"{size:.2f} TB"