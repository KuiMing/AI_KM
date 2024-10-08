"""
Chatbot using RAG (Retrieval Augmented Generation) model.
"""

import os
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from rag import QdrantRAGBot

UPLOAD_FOLDER = "uploads/"
# app config
st.set_page_config(page_title="RAG bot", page_icon="🤖")
st.title("RAG bot")


def main():
    """
    main function for the Streamlit app.
    """
    rag_bot = QdrantRAGBot()
    dataset = [
        name
        for name in os.listdir(UPLOAD_FOLDER)
        if os.path.isdir(os.path.join(UPLOAD_FOLDER, name))
    ]
    dataset_name = st.sidebar.selectbox("請選擇要查詢的資料集名稱", options=dataset)

    # session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello, I am a bot. How can I help you?"),
        ]

    # conversation
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)

    # user input
    user_query = st.chat_input("Type your message here...")
    if user_query is not None and user_query != "":
        st.session_state.chat_history.append(HumanMessage(content=user_query))

        with st.chat_message("Human"):
            st.markdown(user_query)

        with st.chat_message("AI"):
            response = st.write_stream(
                rag_bot.get_response(
                    user_query=user_query,
                    chat_history=st.session_state.chat_history,
                    dataset_name=dataset_name,
                )
            )

        st.session_state.chat_history.append(AIMessage(content=response))


if __name__ == "__main__":
    main()
