import os
from decouple import config
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


if not os.environ.get('OPENAI_API_KEY'):
    os.environ['OPENAI_API_KEY'] = config('OPENAI_API_KEY')


model = init_chat_model("gpt-4.1")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

embedding_dim = len(embeddings.embed_query("hello world"))
index = faiss.IndexFlatL2(embedding_dim)

vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)

file_path = './files/nke-10k-2023.pdf'
loader = PyPDFLoader(file_path)
docs = loader.load()

text_spliter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True
)

all_splits = text_spliter.split_documents(docs)

print(f"Split blog post into {len(all_splits)} sub-documents.")

document_ids = vector_store.add_documents(documents=all_splits)
print(document_ids[:3])
