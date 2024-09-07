from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility #langchain vectors can be used 
from langchain_huggingface import HuggingFaceEmbeddings
from bs4 import BeautifulSoup #Langchain wikiloader can be used 
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import google.generativeai as genai #langchain geminillm can be use 

# FastAPI Initialization
app = FastAPI()

# Initialize  Gemini API key
os.environ["GEMINI_API_KEY"] = "enter your api key "
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# Initialize  connection in me+ilvus
connections.connect("default", host="localhost", port="19530")

# Define the collection schema and create the collection in Milvus (set dim=384)
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),  # Embeddings are 384-dimensional float vectors
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2000)  # Increased max_length to 2000
]
schema = CollectionSchema(fields, "Wiki text embedding collection")
collection_name = "wiki_text_collection"

# Drop the collection if it already exists
if utility.has_collection(collection_name):
    collection = Collection(name=collection_name)
    collection.drop()

# Create the collection with the new schema
collection = Collection(name=collection_name, schema=schema)

# Load HuggingFaceEmbeddings using all-MiniLM-L6-v2
embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Function to scrape Wikipedia content and include both paragraphs and list items
def scrape_wikipedia(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')  # Use html.parser

    # Extract paragraphs (<p> tags)
    paragraphs = [para.get_text().strip() for para in soup.find_all('p') if para.get_text().strip()]

    # Extract list items (<li> tags) to capture bullet points
    list_items = [li.get_text().strip() for li in soup.find_all('li') if li.get_text().strip()]

    # Combine both paragraphs and list items
    content = paragraphs + list_items
    return content  # Return combined paragraphs and list items


# Pydantic model for load request
class LoadRequest(BaseModel):
    wiki_url: str


# Pydantic model for query request
class QueryRequest(BaseModel):
    query: str


# Endpoint for loading data to Milvus
@app.post("/load")
def load_data(request: LoadRequest):
    wiki_url = request.wiki_url
    wiki_content = scrape_wikipedia(wiki_url)

    embeddings = []
    texts = []

    for content in wiki_content:
        # Ensure the content isn't too long for the VARCHAR max_length
        if len(content) > 2000:
            content = content[:2000]

        # Vectorize each paragraph or list item using HuggingFaceEmbeddings
        vector = embeddings_model.embed_documents([content])[0]

        # Add the vector and content to the respective lists
        embeddings.append(vector)
        texts.append(content)

    # Insert the batch of vectors and texts into Milvus
    collection.insert([embeddings, texts])

    # Build an index for efficient vector search
    index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "L2",
        "params": {"nlist": 128}
    }
    collection.create_index(field_name="embedding", index_params=index_params)

    return {"status": "success", "message": f"Data from {wiki_url} has been loaded into Milvus."}


# Endpoint for querying the Milvus collection and enhancing response using Gemini
@app.post("/query")
def query_data(request: QueryRequest):
    query = request.query
    query_vector = embeddings_model.embed_query(query)  # Use embed_query for a single input

    # Ensure the query vector is of length 384
    if len(query_vector) != 384:
        raise HTTPException(status_code=400, detail="Query vector dimension mismatch")

    # Perform search
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    collection.load()

    results = collection.search([query_vector], "embedding", search_params, limit=3, output_fields=["text"])

    # Get the result with the lowest distance
    lowest_distance_result = None
    if results:
        lowest_distance_result = min(results[0], key=lambda hit: hit.distance)
        retrieved_text = lowest_distance_result.entity.get('text')

        # Pass the retrieved text to Gemini LLM for enhancement
        generation_config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain",
        }
        chat_session = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config=generation_config,
            system_instruction="Improve and expand the given text to make it more detailed and informative."
        )
        
        # Chat with Gemini
        response = chat_session.start_chat(history=[{
            "role": "user",
            "parts": [f"\"Text\": \"{retrieved_text}\""]
        }])
        gemini_response = response.send_message(retrieved_text).text

        return {
            "distance": lowest_distance_result.distance,
            "retrieval_response": retrieved_text,
            "gemini_response": gemini_response
        }

    raise HTTPException(status_code=404, detail="No results found")


# Example use cases:

# To load data, send a POST request to /load with a JSON body:
# {
#   "wiki_url": "https://en.wikipedia.org/wiki/Machine_Learning"
# }

# To query data, send a POST request to /query with a JSON body:
# {
#   "query": "name some ML algorithms"
# }
