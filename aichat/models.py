from django.db import models
from rag_app.customs.ModelMixins import TimestampMixin


class UserChats(TimestampMixin):
    user_message = models.TextField()
    ai_message = models.TextField()
    session = models.CharField(blank=True, null=True, max_length=255)
    
    