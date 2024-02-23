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
from ..constants import CSV_FILE, COLLECTION_NAME, MISTRAL_URL, EMBEDDING_FUNCTION

load_dotenv(override=True)


REPLICATE_API_TOKEN = st.secrets["REPLICATE_API_TOKEN"]
sentence_transformer_ef = SentenceTransformerEmbeddingFunction(
    model_name=EMBEDDING_FUNCTION
)


class Bellingcat(Page):
    articles: list[Article]
    replicate_client: Client

    def __init__(self):
        # Initialize the chromadb directory, and client.
        self.chroma_client = chromadb.Client()
        self.replicate_client = Client(api_token=REPLICATE_API_TOKEN)

    def write(self):
        st.title(self.__class__.__name__)

        st.write("## Download CSV")
        df = self.download_csv(CSV_FILE)
        st.dataframe(df.head(2))

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
            relevant_artcles = self.query_collection(prompt)
            if relevant_artcles:
                llm_response = self.query_replicate(prompt, relevant_artcles)
                st.write(llm_response)

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

    def generate_prompt(self, user_prompt: str, relevant_artcles: str) -> str:
        # NOTE: The [INST] and [/INST] tags are required for mistral-7b-instruct to leverage instruction fine-tuning.
        return f"""[INST]
        You are an expert in all things Bellingcat. Your goal is to give me a summary of the top results. You will be given a USER_PROMPT, and a series of RELEVANT_ARTICLES.

        USER_PROMPT: {user_prompt}

        RELEVANT_ARTICLES: {relevant_artcles}

        SUGGESTIONS:

        [/INST]
        """

    def query_replicate(
        self,
        user_prompt: str,
        relevant_artcles: str,
        temperature: float = 0.75,
        max_new_tokens: int = 2048,
    ) -> str:
        # Prompt the mistral-7b-instruct LLM
        mistral_response = self.replicate_client.run(
            MISTRAL_URL,
            input={
                "prompt": self.generate_prompt(user_prompt, relevant_artcles),
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
            },
        )

        # Concatenate the response into a single string.
        return "".join([str(s) for s in mistral_response])
