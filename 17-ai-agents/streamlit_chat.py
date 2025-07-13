import streamlit as st
from agent_core import run_agent

st.set_page_config(page_title="AI Support Agent", page_icon="🤖")
st.title("🛍️ ShopSmart AI Agent")

# Session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "ai", "content": "Hello! How can I assist you today?"})

# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# User input
if user_input := st.chat_input("Type your message..."):
    st.chat_message("user").write(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("ai"):
        response_placeholder = st.empty()
        try:
            response = run_agent(user_input)
        except Exception as e:
            response = f"⚠️ Error: {e}"
        response_placeholder.write(response)
        st.session_state.messages.append({"role": "ai", "content": response})
