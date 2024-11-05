"""
This module contains the code to embed a PDF file into a Qdrant collection.
"""

from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import (
    AzureOpenAIEmbeddings,
    AzureChatOpenAI,
    OpenAIEmbeddings,
    ChatOpenAI,
)
from langchain_qdrant import QdrantVectorStore
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.prompts import ChatPromptTemplate

# pylint: disable=no-name-in-module
from langchain import schema
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import dotenv_values
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Filter,
    FieldCondition,
    MatchValue,
)
import pandas as pd
import camelot


class QdrantRAGBot:
    def __init__(self, config_path: str = ".env"):
        config = dotenv_values(config_path)
        self.qdrant_url = config.get("QDRANT_URL")
        source = config.get("SOURCE")
        if source == "OpenAI":
            self.embedding_llm = OpenAIEmbeddings(
                api_key=config.get("OPENAI_API_KEY"), model="text-embedding-3-large"
            )
            self.generator_llm = ChatOpenAI(
                api_key=config.get("OPENAI_API_KEY"),
                model="gpt-4o",
            )
        else:
            self.embedding_llm = AzureOpenAIEmbeddings(
                azure_endpoint=config.get("AZURE_OPENAI_ENDPOINT"),
                azure_deployment=config.get("AZURE_OPENAI_Embedding_DEPLOYMENT_NAME"),
                api_key=config.get("AZURE_OPENAI_KEY"),
                openai_api_version=config.get("AZURE_OPENAI_API_VERSION"),
            )
            self.generator_llm = AzureChatOpenAI(
                azure_endpoint=config.get("AZURE_OPENAI_ENDPOINT"),
                azure_deployment=config.get("AZURE_OPENAI_DEPLOYMENT_NAME"),
                openai_api_version=config.get("AZURE_OPENAI_API_VERSION"),
                api_key=config.get("AZURE_OPENAI_KEY"),
                streaming=True,
            )

    def embed_pdf(
        self,
        dataset: str,
        pdf_path: str,
        collection: str = "test",
        overwrite: bool = False,
    ):
        """
        Embeds the text from a PDF file into a Qdrant collection.
        Args:
            dataset: The name of the dataset to use for reference.
            pdf_path: The path to the PDF file to embed.
            collection: The name of the Qdrant collection to create.
            overwrite: Whether to overwrite the existing collection with the same name.
        """
        if overwrite:
            client = QdrantClient(url=self.qdrant_url)
            # client.delete_collection(collection)
            filter_condition = Filter(
                must=[
                    FieldCondition(
                        key="metadata.dataset", match=MatchValue(value=dataset)
                    ),
                    FieldCondition(
                        key="metadata.file_name", match=MatchValue(value=pdf_path)
                    ),
                ]
            )
            client.delete(collection_name=collection, points_selector=filter_condition)
            print(f"Deleted file {pdf_path} from collection {collection}")

        qdrant = self.embed_text(dataset, pdf_path, collection)
        qdrant = self.embed_tables(dataset, pdf_path, collection)
        return qdrant

    def embed_text(
        self,
        dataset: str,
        pdf_path: str,
        collection: str = "test",
    ):
        """
        Embeds the text from a PDF file into a Qdrant collection.
        Args:
            dataset: The name of the dataset to use for reference.
            pdf_path: The path to the PDF file to embed.
            collection: The name of the Qdrant collection to create.
        """
        text_splitter = SemanticChunker(
            self.embedding_llm, breakpoint_threshold_type="gradient"
        )
        loader = PyPDFLoader(pdf_path)
        pages = loader.load_and_split()
        try:
            splits = text_splitter.create_documents(
                texts=[page.page_content for page in pages],
                metadatas=[
                    {
                        "file_name": pdf_path,
                        "dataset": dataset,
                    }
                    for page in pages
                ],
            )
        except Exception as e:
            for i in range(len(pages)):
                try:
                    splits = text_splitter.create_documents(
                        texts=[pages[i].page_content],
                        metadatas=[
                            {
                                "file_name": pdf_path,
                                "dataset": dataset,
                            }
                        ],
                    )
                except Exception as e:
                    if i + 2 <= len(pages) - 1:
                        splits = text_splitter.create_documents(
                            texts=[
                                pages[i].page_content
                                + pages[i + 1].page_content
                                + pages[i + 2].page_content
                            ],
                            metadatas=[
                                {
                                    "file_name": pdf_path,
                                    "dataset": dataset,
                                }
                            ],
                        )
                    elif i + 2 == len(pages):
                        splits = text_splitter.create_documents(
                            texts=[
                                pages[i - 1].page_content
                                + pages[i].page_content
                                + pages[i + 1].page_content
                            ],
                            metadatas=[
                                {
                                    "file_name": pdf_path,
                                    "dataset": dataset,
                                }
                            ],
                        )
                    else:
                        splits = text_splitter.create_documents(
                            texts=[
                                pages[i - 2].page_content
                                + pages[i - 1].page_content
                                + pages[i].page_content
                            ],
                            metadatas=[
                                {
                                    "file_name": pdf_path,
                                    "dataset": dataset,
                                }
                            ],
                        )

        qdrant = QdrantVectorStore.from_documents(
            splits,
            embedding=self.embedding_llm,
            url=self.qdrant_url,
            collection_name=collection,
        )

        return qdrant

    def get_response(
        self,
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

        question_answer_chain = create_stuff_documents_chain(self.generator_llm, prompt)
        client = QdrantClient(url=self.qdrant_url)
        qdrant = QdrantVectorStore(
            client=client, collection_name=collection_name, embedding=self.embedding_llm
        )
        retriever = qdrant.as_retriever(
            search_kwargs=dict(
                k=3,
                filter=Filter(
                    must=[
                        FieldCondition(
                            key="metadata.dataset",
                            match=MatchValue(value=dataset_name),
                        )
                    ]
                ),
            )
        )

        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        chain = rag_chain.pick("answer")
        return chain.stream({"input": user_query, "chat_history": chat_history})

    @staticmethod
    def merge_tables(tables):
        merged_tables = []
        merged_df = pd.DataFrame()
        previous_table = None

        def is_same_table(df1, df2):
            return len(df1.columns) == len(df2.columns)

        for table in tables[1:]:
            df = table.df
            if previous_table is not None and is_same_table(previous_table, df):
                # 如果是同一個表格，則合併數據
                df.columns = previous_table.columns
                merged_df = pd.concat([merged_df, df], ignore_index=True)
            else:
                # 如果是新表格，設置新的列標題
                merged_tables.append(merged_df)
                merged_df = pd.DataFrame()
                df.columns = df.iloc[0]
                df = df.drop(0)
                previous_table = df
                merged_df = pd.concat([merged_df, df], ignore_index=True)
        merged_tables.append(merged_df)
        return merged_tables

    def embed_tables(
        self,
        dataset: str,
        pdf_path: str,
        collection: str = "test",
    ):
        """
        Embeds the tables into a Qdrant collection.
        Args:
            dataset: The name of the dataset to use for reference.
            tables: The list of tables to embed.
            collection: The name of the Qdrant collection to create.
            pdf_path: The path to the PDF file to embed.
        """
        tables = camelot.read_pdf(pdf_path, pages="all")
        merged_tables = self.merge_tables(tables)
        texts = []
        for df in merged_tables:
            try:
                texts.append(df.to_json(orient="records"))
            except ValueError:
                pass

        qdrant = QdrantVectorStore.from_texts(
            texts=texts,
            embedding=self.embedding_llm,
            url=self.qdrant_url,
            collection_name=collection,
            metadatas=[{"file_name": pdf_path, "dataset": dataset} for _ in texts],
        )
        return qdrant

    def delete_dataset(
        self,
        dataset: str,
        pdf_path: str,
        collection: str = "test",
    ):
        """
        Deletes the dataset from the Qdrant collection.
        Args:
            dataset_name: The name of the dataset to delete.
        """
        client = QdrantClient(url=self.qdrant_url)
        filter_condition = Filter(
            must=[
                FieldCondition(key="metadata.dataset", match=MatchValue(value=dataset)),
                FieldCondition(
                    key="metadata.file_name", match=MatchValue(value=pdf_path)
                ),
            ]
        )
        client.delete(collection_name=collection, points_selector=filter_condition)
        print(f"Deleted dataset {dataset} from collection")
