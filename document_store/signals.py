from django.db.models.signals import post_save
from django.dispatch import receiver
from aichat.tasks import process_documents
from .models import Document

@receiver(post_save, sender=Document)
def document_embeddings(sender, instance, **kwargs):
    print("document adding for processing.")
    process_documents.delay(instance.id)
    print("document added to processing.")

