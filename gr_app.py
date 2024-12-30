import gradio as gr
import urllib.parse

import chromadb
import os
import time

from collections import defaultdict
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
from langchain.retrievers import MergerRetriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_chroma.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from Callback import StreamlitCallbackHandler

from Generator import Generator

load_dotenv(find_dotenv())

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DATABASE_PATH = "./chroma/"
EMBEDDING_MODEL = "text-embedding-ada-002"


ICON_DELETE = ":material/delete:"
ICON_CHECK = ":material/check:"
ICON_ERROR = ":material/error:"
ICON_WARNING = ":material/warning:"
ICON_INFO = ":material/info:"
ICON_RESTART_ALT = ":material/restart_alt:"

#################################################################################################################################
# Holding Ressources


client = chromadb.PersistentClient(
    path=os.path.join(DATABASE_PATH, f"{EMBEDDING_MODEL}"))

chroma = Chroma(
    collection_name=f"BTW2025",
    client=client,
    create_collection_if_not_exists=False
)

llm = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model="gpt-4o-mini",
    temperature=0.0,
)

embedding = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=OPENAI_API_KEY)

generator = Generator(chroma, embedding, llm)





def chat(message, history):
    response = generator.invoke({"input": message}, {"callbacks": []})
    yield response['answer']

demo = gr.ChatInterface(chat, type="messages")

if __name__ == "__main__":
    demo.launch()
