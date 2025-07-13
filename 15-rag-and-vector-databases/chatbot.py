import os
from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from dotenv import load_dotenv

load_dotenv()

# ----------------------- CONFIGURATION -----------------------
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_CHAT_MODEL = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_EMBEDDING_MODEL = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")

AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
AZURE_SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME", "rag-index")

# ----------------------- SETUP CLIENTS -----------------------
client = AzureOpenAI(
    api_key=AZURE_OPENAI_KEY,
    api_version="2023-05-15",
    azure_endpoint=AZURE_OPENAI_ENDPOINT
)

search_client = SearchClient(
    endpoint=AZURE_SEARCH_ENDPOINT,
    index_name=AZURE_SEARCH_INDEX_NAME,
    credential=AzureKeyCredential(AZURE_SEARCH_KEY)
)

# ----------------------- FUNCTIONS -----------------------
def embed_query(query):
    response = client.embeddings.create(
        input=[query],
        model=AZURE_OPENAI_EMBEDDING_MODEL
    )
    return response.data[0].embedding

def search_similar_docs(query_vector, k=3):
    vector_query = VectorizedQuery(
        vector=query_vector,
        k_nearest_neighbors=k,
        fields="contentVector"
    )

    results = search_client.search(
        search_text=None,  # required fallback text query (even if you use only vector)
        vector_queries=[vector_query],
        select=["content", "path", "chunk_id"]
    )

    return [doc["content"] for doc in results]

def generate_answer(context, question):
    prompt = f"""
You are a helpful assistant. Use the context below to answer the question.

Context:
{context}

Question:
{question}

Answer:"""
    response = client.chat.completions.create(
        model=AZURE_OPENAI_CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return response.choices[0].message.content

# ----------------------- MAIN LOOP -----------------------
if __name__ == "__main__":
    print("💬 Ask me anything based on your ingested documents (type 'exit' to quit)\n")
    while True:
        question = input("❓ You: ")
        if question.strip().lower() == "exit":
            print("👋 Goodbye!")
            break

        print("🔍 Searching documents...")
        query_vector = embed_query(question)
        docs = search_similar_docs(query_vector, k=3)
        context = "\n---\n".join(docs)

        print("🤖 Generating answer...\n")
        answer = generate_answer(context, question)
        print(f"💡 Answer: {answer}\n")
        print("-" * 50)
