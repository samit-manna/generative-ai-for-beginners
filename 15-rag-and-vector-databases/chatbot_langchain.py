from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.azuresearch import AzureSearch
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv

# ------------------ CONFIG ------------------

load_dotenv()
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")
AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT")

AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
AZURE_SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME", "rag-index")

# ------------------ Embedding & Vector Store ------------------
embedding_model = OpenAIEmbeddings(
    deployment=AZURE_EMBEDDING_DEPLOYMENT,
    openai_api_key=AZURE_OPENAI_KEY,
    openai_api_base=AZURE_OPENAI_ENDPOINT,
    openai_api_type="azure",
    openai_api_version="2023-05-15",
)

vectorstore = AzureSearch(
    azure_search_endpoint=AZURE_SEARCH_ENDPOINT,
    azure_search_key=AZURE_SEARCH_KEY,
    index_name=AZURE_SEARCH_INDEX_NAME,
    embedding_function=embedding_model.embed_query,
    embedding_dimensions=1536,
    embedding_field_name="contentVector",
    content_field_name="content"
)

# ------------------ LLM Model ------------------
llm = AzureChatOpenAI(
    openai_api_base=AZURE_OPENAI_ENDPOINT,
    openai_api_key=AZURE_OPENAI_KEY,
    deployment_name=AZURE_DEPLOYMENT_NAME,
    openai_api_type="azure",
    openai_api_version="2023-05-15",
    temperature=0
)

# ------------------ RAG Chain ------------------
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)

# ------------------ Interactive Terminal Chat ------------------
def run_chat():
    print("💬 Ask me anything based on your ingested documents (type 'exit' to quit)\n")
    while True:
        query = input("❓ You: ")
        if query.strip().lower() == "exit":
            print("👋 Goodbye!")
            break
        result = qa_chain(query)
        print(f"\n💡 Answer: {result['result']}\n")
        print("-" * 50)

run_chat()
