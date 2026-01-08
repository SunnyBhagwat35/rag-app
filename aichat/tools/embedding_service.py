from langchain_chroma import Chroma

class EmbeddingService:
    def __init__(self, embedding_model="text-embedding-3-large"):
        self.model = embedding_model
        
    
    def generate_embeddings(self, chunks):
        vector_store = Chroma(
            collection_name="rag_app_collection",
            embedding_function=self.model,
            persist_directory="../../chroma_langchain_db",  # using chroma db to save data locally though we can use postgresql too
        )
        
        try:
            vector_store.add_documents(documents=chunks)
            print("sone embedding")
        except Exception as e:
            print(e)
            print("Something went wrnog while")