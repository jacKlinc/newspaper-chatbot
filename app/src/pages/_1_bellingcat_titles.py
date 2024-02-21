import json
import os
from typing import Any

# fixes chromadb issue
__import__("pysqlite3")
import sys

sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import streamlit as st
import pandas as pd
from replicate import Client
from dotenv import load_dotenv
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from ..types import Page, Article
from ..constants import CSV_FILE, COLLECTION_NAME

load_dotenv(override=True)


REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
sentence_transformer_ef = SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)


class Bellingcat(Page):
    articles: list[Article]
    replicate_client: Client

    def __init__(self):
        # Initialize the chromadb directory, and client.
        self.chroma_client = chromadb.Client()
        self.replicate_client = Client(
            api_token=REPLICATE_API_TOKEN, embedding_function=sentence_transformer_ef
        )

    def write(self):
        st.title(self.__class__.__name__)

        st.write("## Download CSV")
        df = self.download_csv(CSV_FILE)
        st.dataframe(df)

        st.write("## Convert to JSON")
        self.convert_to_json(df)
        st.dataframe(self.articles[0])

        st.write("## Create Vector Database w ChromaDB")
        self.collection = self.chroma_client.get_or_create_collection(
            name=COLLECTION_NAME
        )
        self.fill_chroma_collection()

        prompt = st.chat_input("Enter prompt...")
        if prompt:
            result = self.query_collection(prompt)
            st.write(result)

    def download_csv(self, csv_file: str) -> pd.DataFrame:
        df = pd.read_csv(csv_file)
        df.drop(columns=["year", "month", "path"], inplace=True)
        df = df[["publish_date", "title", "url", "articles_text"]]
        df["id"] = df.index
        return df

    def convert_to_json(self, df: pd.DataFrame) -> None:
        self.articles = json.loads(df.to_json(orient="records"))

    def fill_chroma_collection(self, batch_size: int = 250) -> None:
        for i in range(0, len(self.articles), batch_size):
            batch = self.articles[i : i + batch_size]

            batch_titles = [story["title"] for story in batch]

            # Upsert all of the embeddings, ids, metadata, and title strings into Chromadb.
            self.collection.upsert(
                ids=[str(story["id"]) for story in batch],
                metadatas=[dict(time=story["publish_date"]) for story in batch],
                documents=batch_titles,
                embeddings=sentence_transformer_ef(batch_titles),
            )

    def query_collection(self, query: str, n_results: int = 10) -> str | None:
        results = self.collection.query(query_texts=[query], n_results=n_results)

        if results:
            return "\n".join(results["documents"][0])
