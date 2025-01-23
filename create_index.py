# %%
import chromadb
import os
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_unstructured.document_loaders import UnstructuredLoader
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import MergerRetriever
from langchain.retrievers.document_compressors.flashrank_rerank import FlashrankRerank
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
from unstructured.chunking.basic import chunk_elements
from unstructured.documents.elements import Image

load_dotenv(find_dotenv())

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DATABASE_PATH = "./chroma/"
EMBEDDING_MODEL = "text-embedding-ada-002"

def pretty_output(chunks, mode: str):
    if mode == "elements":
        for i, chunk in enumerate(chunks, 1):
            print(f"Chunk {i}:")
            print(chunk.text)
            print("-" * 120)
            
    elif mode == "documents":
        for i, chunk in enumerate(chunks, 1):
            print(f"Chunk {i}:")
            print(chunk.page_content)
            print("-" * 120)

# %%
docs = {
    "BSW": "BSW_Wahlprogramm_2025.pdf",
    "Grüne": "Grüne_BTW2025.pdf",
    "CDU": "CDU_BTW2025.pdf",
    "AfD": "AfD_Leitantrag-Bundestagswahlprogramm-2025.pdf",
    "Linke": "btw_2025_wahlprogramm_die_linke.pdf",
    "SPD": "BTW_2025_SPD_Regierungsprogramm.pdf",
    "FDP": "fdp-wahlprogramm_2025.pdf",
    "Volt": "volt-programm-bundestagswahl-2025.pdf"
}

# %%
from os import path

# Chunker 2
max_characters = 5000
new_after_n_chars = 1500
overlap = 1000
combine_text_under_n_chars_multiplier=int(new_after_n_chars*(2/3))

DOCS = []

for (party, fpath) in docs.items():
    chunks = UnstructuredLoader(
        file_path=path.join("files", fpath),
        languages=["deu"],
        chunking_strategy="by_title",
        max_characters=max_characters,
        overlap=overlap,
        overlap_all=True,
        combine_text_under_n_chars=combine_text_under_n_chars_multiplier,
        new_after_n_chars=new_after_n_chars,
    ).load()
    for chunk in chunks:
        chunk.metadata["party"] = party
    #print(len(chunks), chunks[0])
    DOCS += chunks


# %%
len(DOCS), DOCS[-1]

# %%
chromadb.configure(allow_reset=True)

# %%
client = chromadb.PersistentClient(
    path=os.path.join(DATABASE_PATH, f"{EMBEDDING_MODEL}")
)

# %%
for chunk in DOCS:
    for md in chunk.metadata:
        if isinstance(chunk.metadata[md], list):
            chunk.metadata[md] = str(chunk.metadata[md])

# %%
Chroma.from_documents(
    documents=DOCS,
    embedding=OpenAIEmbeddings(api_key=OPENAI_API_KEY, model=EMBEDDING_MODEL),
    client=client,
    collection_name=f"BTW2025",
)

# %%



