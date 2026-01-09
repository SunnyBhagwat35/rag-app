import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from rag_app.settings import BASE_DIR

class EmbeddingService:
    def __init__(self, embedding_model="text-embedding-3-large"):
        self.model = embedding_model
        
    
    def generate_embeddings(self, chunks):
        
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        
        vector_store = Chroma(
            collection_name="rag_app_collection",
            embedding_function=embeddings,
            persist_directory=os.path.join(BASE_DIR, 'chroma_langchain_db'),  # using chroma db to save data locally though we can use postgresql too
        )
        
        try:
            vector_store.add_documents(documents=chunks)
            print("sone embedding")
        except Exception as e:
            print(e)
            print("Something went wrnog while")