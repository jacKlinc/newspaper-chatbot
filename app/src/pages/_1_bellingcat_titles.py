import json
import urllib.request

# fixes chromadb issue
__import__("pysqlite3")
import sys

sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import streamlit as st
from replicate import Client
from dotenv import load_dotenv
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


from ..types import Page, Article
from ..constants import JSON_FILE, COLLECTION_NAME, MISTRAL_URL, EMBEDDING_FUNCTION

load_dotenv(override=True)


REPLICATE_API_TOKEN = st.secrets["REPLICATE_API_TOKEN"]
sentence_transformer_ef = SentenceTransformerEmbeddingFunction(
    model_name=EMBEDDING_FUNCTION
)
chroma_client = chromadb.Client()


class Bellingcat(Page):
    articles: list[Article]
    replicate_client: Client

    def __init__(self):
        # Initialize the chromadb directory, and client.
        self.replicate_client = Client(api_token=REPLICATE_API_TOKEN)

    def write(self):
        st.title(self.__class__.__name__)
        st.subheader("An open souce researcher's personal assistant")
        self.articles = download_json()
        self.collection = fill_chroma_collection(self.articles)

        # Chatbot
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # React to user input
        if prompt := st.chat_input("Enter prompt..."):
            # Display user message in chat message container
            st.chat_message("user").markdown(prompt)
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})

            # find relevant articles
            relevant_artcles = self.query_collection(prompt)
            if relevant_artcles:
                # query LLM
                llm_response = self.query_replicate(prompt, relevant_artcles)
                st.chat_message("").markdown(llm_response)

            # Add assistant response to chat history
            st.session_state.messages.append(
                {"role": "assistant", "content": f"Echo: {prompt}"}
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


@st.cache_data
def download_json():
    with urllib.request.urlopen(JSON_FILE) as url:
        return json.loads(url.read().decode("utf8"))


@st.cache_resource
def fill_chroma_collection(articles: list[Article], batch_size: int = 250):
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
    for i in range(0, len(articles), batch_size):
        batch = articles[i : i + batch_size]

        batch_titles = [story["title"] for story in batch]

        # Upsert all of the embeddings, ids, metadata, and title strings into Chromadb.
        collection.upsert(
            ids=[str(story["id"]) for story in batch],
            metadatas=[dict(time=story["publish_date"]) for story in batch],
            documents=batch_titles,
            embeddings=sentence_transformer_ef(batch_titles),
        )
    return collection
