"""
This module contains the code to embed a PDF file into a Qdrant collection.
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import AzureOpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_experimental.text_splitter import SemanticChunker
from dotenv import dotenv_values
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Filter,
    FieldCondition,
    MatchValue,
)


def embed_pdf(
    dataset: str, pdf_path: str, collection: str = "test", overwrite: bool = False
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
        client = QdrantClient(url="http://localhost:6333")
        # client.delete_collection(collection)
        filter_condition = Filter(
            must=[
                FieldCondition(key="metadata.dataset", match=MatchValue(value=dataset)),
                FieldCondition(
                    key="metadata.file_name", match=MatchValue(value=pdf_path)
                ),
            ]
        )
        client.delete(collection_name=collection, points_selector=filter_condition)
        print(f"Deleted file {pdf_path} from collection {collection}")

    config = dotenv_values(".env")

    embedding_llm = AzureOpenAIEmbeddings(
        azure_endpoint=config.get("AZURE_OPENAI_ENDPOINT"),
        azure_deployment=config.get("AZURE_OPENAI_Embedding_DEPLOYMENT_NAME"),
        api_key=config.get("AZURE_OPENAI_KEY"),
        openai_api_version=config.get("AZURE_OPENAI_API_VERSION"),
    )

    text_splitter = SemanticChunker(embedding_llm, breakpoint_threshold_type="gradient")

    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()
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

    qdrant = QdrantVectorStore.from_documents(
        splits,
        embedding=embedding_llm,
        url="http://localhost:6333",
        collection_name=collection,
    )

    return qdrant
