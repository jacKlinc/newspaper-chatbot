import json
import os

import streamlit as st
import pandas as pd
from replicate import Client
from dotenv import load_dotenv

from ..types import Page, Article
from ..constants import CSV_FILE, EMBEDDINGS_MODEL, COLLECTION_NAME

load_dotenv(override=True)

REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")


class Bellingcat(Page):
    replicate_client = Client(api_token=REPLICATE_API_TOKEN)

    def write(self):
        st.title(self.__class__.__name__)

        st.write("## Download CSV")
        df = self.download_csv(CSV_FILE)
        st.dataframe(df)

        st.write("## Convert to JSON")
        articles = self.convert_to_json(df)
        st.dataframe(articles[0])

    def download_csv(self, csv_file: str) -> pd.DataFrame:
        df = pd.read_csv(csv_file)
        df.drop(columns=["year", "month", "path"], inplace=True)
        df = df[["publish_date", "title", "url", "articles_text"]]
        df["id"] = df.index
        return df

    def convert_to_json(self, df: pd.DataFrame) -> list[Article]:
        return json.loads(df.to_json(orient="records"))
