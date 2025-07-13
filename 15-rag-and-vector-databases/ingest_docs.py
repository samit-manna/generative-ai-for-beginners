import os
import openai
from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    VectorSearchAlgorithmConfiguration,
    VectorSearchAlgorithmKind,
    VectorSearchProfile,
    HnswAlgorithmConfiguration,
    HnswParameters
)
import glob
import uuid
from dotenv import load_dotenv

# ----------------------- CONFIGURATION -----------------------
load_dotenv()
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")

AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
AZURE_SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME", "rag-index")

# ----------------------- SETUP CLIENTS -----------------------
openai.api_type = "azure"
openai.api_base = AZURE_OPENAI_ENDPOINT
openai.api_version = "2023-05-15"
openai.api_key = AZURE_OPENAI_KEY

index_client = SearchIndexClient(
    endpoint=AZURE_SEARCH_ENDPOINT,
    credential=AzureKeyCredential(AZURE_SEARCH_KEY)
)
search_client = SearchClient(
    endpoint=AZURE_SEARCH_ENDPOINT,
    index_name=AZURE_SEARCH_INDEX_NAME,
    credential=AzureKeyCredential(AZURE_SEARCH_KEY)
)

client = AzureOpenAI(
    api_key=AZURE_OPENAI_KEY,
    api_version="2023-05-15",
    azure_endpoint=AZURE_OPENAI_ENDPOINT
)

# ----------------------- INDEX CREATION -----------------------
def create_index():
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True, sortable=True, filterable=True, facetable=True),
        SearchableField(name="content", type=SearchFieldDataType.String),
        SearchField(
            name="contentVector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=1536,  # OpenAI embeddings dimension
            vector_search_profile_name="myHnswProfile"  # Reference to the vector search profile
        ),
        SimpleField(name="path", type=SearchFieldDataType.String, sortable=True, filterable=True, facetable=True),
        SimpleField(name="chunk_id", type=SearchFieldDataType.String, sortable=True, filterable=True, facetable=True),
    ]

    # Update the vector search configuration
    vector_search = VectorSearch(
        profiles=[
            VectorSearchProfile(
                name="myHnswProfile",
                algorithm_configuration_name="myHnsw"
            )
        ],
        algorithms=[
            HnswAlgorithmConfiguration(
                name="myHnsw",
                parameters=HnswParameters(
                    m=4,
                    ef_construction=400,
                    ef_search=500,
                    metric="cosine"
                )
            )
        ]
    )

    index = SearchIndex(
        name=AZURE_SEARCH_INDEX_NAME,
        fields=fields,
        vector_search=vector_search
    )

    try:
        index_client.create_index(index)
        print("✅ Index created.")
    except Exception as e:
        print("⚠️ Could not create index (may already exist):", e)

# ----------------------- EMBEDDING -----------------------
def embed_text(text):
    response = client.embeddings.create(
        input=text,
        model=AZURE_OPENAI_EMBEDDING_DEPLOYMENT
    )
    return response.data[0].embedding

# ----------------------- LOAD .MD FILES -----------------------
def load_documents_from_folder(folder_path):
    docs = []
    print(f"📂 Loading documents from folder: {folder_path}")
    md_files = glob.glob(f"{folder_path}/*.md")
    print(f"Found {len(md_files)} markdown files.")
    for file_path in md_files:
        print(f"Loading file: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            # Chunking: split every 1000 characters
            chunks = [content[i:i + 1000] for i in range(0, len(content), 1000)]
            for i, chunk in enumerate(chunks):
                filename = os.path.basename(file_path).replace(".", "_")
                doc_id = f"{filename}-{i}-{uuid.uuid4()}"
                docs.append({
                    "id": doc_id,
                    "content": chunk
                })
    return docs

# ----------------------- INGEST -----------------------
def ingest_documents(docs):
    print(f"📥 Ingesting {len(docs)} documents...")
    actions = []
    for doc in docs:
        print(f"Processing document: {doc['id']}")
        embedding = embed_text(doc["content"])
        actions.append({
            "id": doc["id"],
            "content": doc["content"],
            "contentVector": embedding
        })
    result = search_client.upload_documents(documents=actions)
    print(f"📥 Uploaded {len(result)} documents.")

# ----------------------- MAIN -----------------------
if __name__ == "__main__":
    # create_index()
    documents = load_documents_from_folder("15-rag-and-vector-databases/data")
    ingest_documents(documents)
