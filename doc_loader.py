import os
from decouple import config
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS


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

