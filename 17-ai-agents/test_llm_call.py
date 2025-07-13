import os
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

# Raw diagnostic with openai SDK
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2023-12-01-preview"  # update if needed
)

try:
    response = client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),  # this must be the deployment name
        messages=[{"role": "user", "content": "Hello, how are you?"}],
        max_tokens=50,
    )
    print("✅ Success:", response.choices[0].message.content)
except Exception as e:
    print("❌ Failed:", e)
