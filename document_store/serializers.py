from rest_framework import serializers
from .models import Document


class FileUploadSerializer(serializers.ModelSerializer):
    """
    Serializer for file upload - handles file validation and creation
    """
    file_size_display = serializers.SerializerMethodField(read_only=True)
    file_url = serializers.SerializerMethodField(read_only=True)
    
    class Meta:
        model = Document
        fields = [
            'id',
            'file',
            'original_name',
            'file_size',
            'file_size_display',
            'file_type',
            'description',
            'uploaded_at',
            'file_url'
        ]
        read_only_fields = ['id', 'original_name', 'file_size', 'file_type', 'uploaded_at']
        extra_kwargs = {
            'file': {'write_only': True}
        }
    
    def get_file_size_display(self, obj):
        return obj.get_file_size_display()
    
    def get_file_url(self, obj):
        if obj.file:
            request = self.context.get('request')
            if request:
                return request.build_absolute_uri(obj.file.url)
            return obj.file.url
        return None
    
    def validate_file(self, value):
        """
        Validate file size and type if needed
        """
        # Check file size (10MB limit)
        max_size = 10 * 1024 * 1024  # 10MB
        if value.size > max_size:
            raise serializers.ValidationError("File size exceeds 10MB limit.")
        
        # You can add file type validation here if needed
        # allowed_types = ['.pdf', '.doc', '.docx', '.txt', '.jpg', '.png']
        # file_extension = os.path.splitext(value.name)[1].lower()
        # if file_extension not in allowed_types:
        #     raise serializers.ValidationError(f"File type {file_extension} is not allowed.")
        
        return value


class FileListSerializer(serializers.ModelSerializer):
    """
    Serializer for listing files - optimized for list view
    """
    file_size_display = serializers.SerializerMethodField()
    file_url = serializers.SerializerMethodField()
    
    class Meta:
        model = Document
        fields = [
            'id',
            'original_name',
            'file_size',
            'file_size_display',
            'file_type',
            'description',
            'uploaded_at',
            'file_url'
        ]
    
    def get_file_size_display(self, obj):
        return obj.get_file_size_display()
    
    def get_file_url(self, obj):
        if obj.file:
            request = self.context.get('request')
            if request:
                return request.build_absolute_uri(obj.file.url)
            return obj.file.url
        return None