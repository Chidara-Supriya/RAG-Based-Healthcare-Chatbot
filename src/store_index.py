from dotenv import load_dotenv
import os
from src.helper import load_pdf_file, filter_to_minimal, text_split,dowload_hugging_face_embeddings
from pinecone import Pinecone
from pinecone import ServerlessSpec

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

extracted_Data = load_pdf_file(data = 'data/')
filter_data = filter_to_minimal_docs(extracted_data)
text_chunks = text_split(filter_data)

embedding = dowload_embeddings()
from pinecone import Pinecone
pinecone_api_key = PINECONE_API_KEY

pc = Pinecone(api_key = pinecone_api_key)

index_name = "healthcarebot"

if not pc.has_index(index_name):
    pc.create_index(
        name = index_name,
        dimension = 384,
        metric = "cosine",
        spec = ServerlessSpec(cloud="aws",region="us-east-1")
    )

index = pc.Index(index_name)

docsearch = PineconeVectorStore.from_documents(
    documents = text_chunks,
    index_name = index_name,
    embedding = embeddings,

)

















# Load Existing Index

from langchain_pinecone import PineconeVectorStore

# Embed each chunk and upset the embeddings into your Pinecone index
docsearch = PineconeVectorStore.from_existing_index(
    index_name = index_name,
    embedding = embedding
)