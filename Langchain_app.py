from fastapi import FastAPI
import uvicorn
from pymilvus import (
    connections, FieldSchema, CollectionSchema, DataType, Collection, utility
)
import os
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

app = FastAPI()

# Environment variables and connection setup
inference_api_key = "add your API key"
google_api_key = os.getenv("GEMINI_API_KEY")

# Initialize Milvus connection
connections.connect("default", host="localhost", port="19530")

# Define Milvus collection schema
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2000)
]
schema = CollectionSchema(fields, description="ML text and embeddings")
collection_name = "ml_text_embeddings"

# Drop and create the collection
if utility.has_collection(collection_name):
    collection = Collection(name=collection_name)
    collection.drop()
collection = Collection(name=collection_name, schema=schema)

# Fetch and parse data using BeautifulSoup
url = "https://en.wikipedia.org/wiki/Karate"
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')
text = soup.get_text()

# Split the text using RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
split_text = text_splitter.split_text(text)

# Initialize the HuggingFace Inference API Embeddings model
embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=inference_api_key, model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Prepare and insert data into Milvus
milvus_embeddings = [embeddings.embed_query(chunk) for chunk in split_text]
milvus_texts = [chunk[:2000] for chunk in split_text]
entities = [milvus_embeddings, milvus_texts]
collection.insert(entities)

# Create an index and load the collection
index_params = {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 128}}
collection.create_index(field_name="embedding", index_params=index_params)
collection.load()

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    google_api_key=google_api_key
)

# FastAPI endpoints
@app.post("/load")
def load_collection():
    collection.load()
    return {"message": "Collection loaded successfully"}

@app.post("/query")
async def perform_query(query: str, k: int = 1):
    query_embedding = embeddings.embed_query(query)
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    results = collection.search([query_embedding], "embedding", search_params, limit=k, output_fields=["text"])
    top_result = results[0][0] if results and results[0] else None

    if top_result:
        retrieved_output = top_result.entity.get('text')
        distance = top_result.distance
        messages = [
            {"role": "system", "content": "You are a helpful assistant that enhances text content."},
            {"role": "user", "content": retrieved_output}
        ]
        ai_msg = llm.invoke(messages)
        return {"query": query, "distance": distance, "retrieved_output": retrieved_output, "enhanced_output": ai_msg.content}
    else:
        return {"message": f"No results found for query: '{query}'"}

# Run the server
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
