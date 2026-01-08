import os
from celery import shared_task
from document_store.models import Document
from .tools.doc_loader import DocumentLoader
from .tools.doc_splitter import TextSplitter
from .tools.embedding_service import EmbeddingService


@shared_task(bind=True)
def process_documents(self, document_id):
    try:
        docuemnt:Document = Document.objects.get(pk=document_id)
    except Exception as e:
        print('Document not found.')
        return {'error': "Document not found."}

    loader = DocumentLoader()
    doc = loader.load_pdf(docuemnt.file.path)
    # if doc['error']:
    #    return {}
   
    splitter = TextSplitter()
    chunks = splitter.get_doc_splits(doc)
    
    embedder = EmbeddingService()
    embedder.generate_embeddings(chunks)
    
    print("Document processed")
    return {"message": f"docuemnt processed: {docuemnt.id} | {docuemnt.original_name}"}
        
        