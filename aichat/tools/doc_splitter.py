from langchain_text_splitters import RecursiveCharacterTextSplitter


class TextSplitter:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, *args, **kwargs):

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
    def get_doc_splits(self, docs):
        """Splits documents into small chunks

        Args:
            docs: Loaded docs by langchain
        """
        text_spliter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            add_start_index=True
        )
        all_splits = text_spliter.split_documents(docs)
        print("done splittings")
        return all_splits
        
        