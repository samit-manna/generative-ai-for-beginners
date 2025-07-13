import os
from dotenv import load_dotenv
from langchain.chat_models import AzureChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain.memory import ConversationBufferMemory

load_dotenv()

llm = AzureChatOpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    openai_api_version="2023-12-01-preview",
    temperature=0.7,
    streaming=True,  # enable for real-time Streamlit updates
)

def get_faq_answer(question: str) -> str:
    faqs = {
        "shipping": "We offer free shipping for orders over ₹500. Delivery time is 3-5 business days.",
        "return": "Returns are accepted within 15 days. Please ensure the product is unused.",
        "warranty": "All electronics come with a 1-year warranty.",
    }
    for key in faqs:
        if key in question.lower():
            return faqs[key]
    return "I'm not sure about that. Let me escalate it to a human for you."

faq_tool = Tool(
    name="FAQTool",
    func=get_faq_answer,
    description="Use this to answer questions about shipping, returns, and warranties."
)

memory = ConversationBufferMemory(memory_key="chat_history")

agent = initialize_agent(
    tools=[faq_tool],
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    memory=memory,
    verbose=True,
)

def run_agent(query):
    return agent.run(query)
