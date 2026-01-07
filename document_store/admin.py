from django.contrib import admin
from .models import Document


@admin.register(Document)
class UploadedFileAdmin(admin.ModelAdmin):
    list_display = ['original_name', 'file_type', 'get_file_size_display', 'uploaded_at']
    list_filter = ['file_type', 'uploaded_at']
    search_fields = ['original_name', 'description']
    readonly_fields = ['original_name', 'file_size', 'file_type', 'uploaded_at']
    ordering = ['-uploaded_at']
    
    def get_file_size_display(self, obj):
        return obj.get_file_size_display()
    get_file_size_display.short_description = 'File Size'
