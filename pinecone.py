pip install pypdf
import pinecone
from langchain_community.vectorstores import Pinecone
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
from pinecone import Pinecone
import os
os.environ[
    "PINECONE_API_KEY"
] = "your_pc_key"
os.environ["OPENAI_API_KEY"] = "your_api_key"

api_key = os.environ["PINECONE_API_KEY"]

pc = Pinecone(api_key=api_key)
from langchain.globals import set_debug

set_debug(True)
from langchain_openai import OpenAIEmbeddings

pc = Pinecone(api_key="your_pc_key")
pinecone_index = pc.Index("your_index_name")
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
file = PyPDFLoader("C:/Users/ASUS/OneDrive/Desktop/dataset/company policy.pdf")
documents = file.load()
text_splitter = CharacterTextSplitter(chunk_size=1000,chunk_overlap=0)
docs = text_splitter.split_documents(documents)
from langchain_pinecone import Pinecone
index_name="hrbot"
embeddings = OpenAIEmbeddings(model = "text-embedding-ada-002",api_key="your_api_key")
docsearch = Pinecone.from_documents(docs,embeddings,index_name=index_name)
from langchain.vectorstores import Pinecone

text_field = "text"


index = pc.Index(index_name)

vectorstore = Pinecone(
    index, embeddings.embed_query, text_field
)
from langchain.vectorstores import Pinecone
query = "What are some company policies?"
#when the vectorstore is called for similarity_search,with arg query, the query will be embedded by embed_query
vectorstore.similarity_search(
    query,  # our search query
    k=3  # return 3 most relevant docs
)
