
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import streamlit as st
import urllib.parse

import chromadb
import os

from collections import defaultdict
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
from langchain.retrievers import MergerRetriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_chroma.vectorstores import Chroma

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


@st.cache_resource
def getClient():
    return chromadb.PersistentClient(
        path=os.path.join(DATABASE_PATH, f"{EMBEDDING_MODEL}"),
    )


@st.cache_resource
def getChroma():
    return Chroma(
        collection_name=f"BTW2025",
        client=getClient(),
        create_collection_if_not_exists=False
    )


@st.cache_resource
def getLLM():
    return ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model="gpt-4o-mini",
        temperature=0.0,
    )


@st.cache_resource
def getEmbedding():
    return OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=OPENAI_API_KEY)


@st.cache_resource
def getGenerator():
    return Generator(getChroma(), getEmbedding(), getLLM())


if "messages_history" not in st.session_state:
    st.session_state.messages_history = [
        {"role": "ai", "content": """Guten Tag! Ich habe die Wahlprogramme der Parteien zur Bundestagswahl 2025 gelesen und beantworte gerne Deine Fragen dazu! 
                                     Womit darf ich Dir behilflich sein?"""}
    ]


def clear_chat_history():
    try:
        st.session_state.messages_history = [
            {"role": "ai", "content": "Guten Tag! Ich bin der digitale Assistent dieser Einrichtung und helfe Ihnen gerne weiter. Womit darf ich Ihnen behilflich sein??"}
        ]
        st.success("Chat History Cleared Successfully", icon=ICON_CHECK)
    except:
        st.error("An Error Occured while clearing the Chat History",
                 icon=ICON_ERROR)


#################################################################################################################################
st.title("ChatBot")


tab_chatbot, tab_chunks = st.tabs(["ChatBot", "Chunks"])

query = st.chat_input("Type your message here...")

with tab_chatbot:
    for message in st.session_state.messages_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if query:
        try:
            st.session_state.messages_history.append(
                {"role": "user", "content": query})
            st.chat_message("user").write(query)

            generator = getGenerator()
            with st.chat_message("ai"):
                st_callback = StreamlitCallbackHandler(st.container())
                response_generator = generator.invoke({"input": query}, {"callbacks": [st_callback]})
                response_str = st.write(response_generator)
                st.session_state.messages_history.append(
                    {"role": "ai", "content": response_str})
        except Exception as e:
            st.error(
                f"An Error Occured: {e}. Clear cache and try again.", icon=ICON_ERROR)


def transform_source_to_link_md(file_path):
    formatted_path = file_path.replace("\\", "/").replace("Data", "Source")
    encoded_path = urllib.parse.quote(formatted_path)
    file_url = f"file:///{encoded_path}"
    return f"[{os.path.basename(file_path)}]({file_url})"


with tab_chunks:
    generator = getGenerator()
    context = generator.context

    if context:
        chunk_selection = st.selectbox(
            "Chunk",
            options=[i for i in range(len(context))],
            format_func=lambda x: x+1,
        )
        chunk = context[chunk_selection]
        content = chunk.page_content
        source = transform_source_to_link_md(chunk.metadata["source"])
        page_number = chunk.metadata["page_number"]

        st.markdown(f"""#### Source:
{source}""")
        st.markdown(f"""#### Page Number:
{page_number}""")
        st.markdown(f"""#### Content:
{content}""")
    else:
        st.info("Please type a message in the ChatBot Tab to see the Chunks. You can only see the Chunks from the last query.", icon=ICON_INFO)
