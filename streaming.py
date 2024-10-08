"""
Chatbot using RAG (Retrieval Augmented Generation) model.
"""

from typing import List
import os
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI

# pylint: disable=no-name-in-module
from langchain import schema
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from langchain_qdrant import QdrantVectorStore
from dotenv import dotenv_values


config = dotenv_values(".env")

UPLOAD_FOLDER = "uploads/"
# app config
st.set_page_config(page_title="RAG bot", page_icon="🤖")
st.title("RAG bot")


def get_response(
    user_query: str,
    chat_history: List[schema.HumanMessage],
    dataset_name: str,
    collection_name: str = "test",
):
    """
    Generates a response to the user's query based on the provided
    chat history and a specified dataset.
    Args:
        user_query: The query from the user.
        chat_history: The history of the chat as a list of HumanMessage objects.
        dataset_name: The name of the dataset to use for reference.
        collection_name: The name of the Qdrant collection to use for retrieval.
    Returns:
        generator: A generator that streams the response to the user's query.
    Raises:
        ValueError: If any required configuration is missing or invalid.
    """
    system_prompt = (
        "你是一位專門根據文件回答問題的 AI 助手。如果你無法從文件得到答案，請說你不知道。"
        "請根據以下參考資料回答問題："
        "歷史紀錄：{chat_history}"
        "參考資料：{context}"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    generator_llm = AzureChatOpenAI(
        azure_endpoint=config.get("AZURE_OPENAI_ENDPOINT"),
        azure_deployment=config.get("AZURE_OPENAI_DEPLOYMENT_NAME"),
        openai_api_version=config.get("AZURE_OPENAI_API_VERSION"),
        api_key=config.get("AZURE_OPENAI_KEY"),
        streaming=True,
    )
    embedding_llm = AzureOpenAIEmbeddings(
        azure_endpoint=config.get("AZURE_OPENAI_ENDPOINT"),
        azure_deployment=config.get("AZURE_OPENAI_Embedding_DEPLOYMENT_NAME"),
        api_key=config.get("AZURE_OPENAI_KEY"),
        openai_api_version=config.get("AZURE_OPENAI_API_VERSION"),
    )
    question_answer_chain = create_stuff_documents_chain(generator_llm, prompt)
    client = QdrantClient(url="http://localhost:6333")
    qdrant = QdrantVectorStore(
        client=client, collection_name=collection_name, embedding=embedding_llm
    )
    retriever = qdrant.as_retriever(
        search_kwargs=dict(
            k=3,
            filter=qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="metadata.dataset",
                        match=qdrant_models.MatchValue(value=dataset_name),
                    )
                ]
            ),
        )
    )

    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    chain = rag_chain.pick("answer")
    return chain.stream({"input": user_query, "chat_history": chat_history})


def main():
    """
    main function for the Streamlit app.
    """

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
                get_response(
                    user_query=user_query,
                    chat_history=st.session_state.chat_history,
                    dataset_name=dataset_name,
                )
            )

        st.session_state.chat_history.append(AIMessage(content=response))


if __name__ == "__main__":
    main()
