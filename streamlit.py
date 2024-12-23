import streamlit as st
import urllib.parse

import chromadb
import os

from collections import defaultdict
from typing import List
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.embeddings.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.retrievers import MergerRetriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

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

docs = {
    "BSW": "BSW_Parteiprogramm.pdf",
    "Grüne": "Grüne_BTW2025.pdf",
    "CDU": "CDU_BTW2025.pdf",
    "AfD": "Programm_AfD_Online_.pdf",
    "Linke": "DIE_LINKE_Wahlprogramm_zur_Bundestagswahl_2021.pdf",
    "SPD": "SPD-Zukunftsprogramm.pdf",
    "FDP": "FDP_Programm_Bundestagswahl2021_1.pdf"
}


@st.cache_resource
def getClient():
    return chromadb.PersistentClient(
        path=os.path.join(DATABASE_PATH, f"{EMBEDDING_MODEL}"),
    )

class PrettyOutput:
    
    @staticmethod
    def output_per_line(text: str, words_per_line: int=10) -> str:
        text_parts = text.split('\n')
        pretty_text = ''
        
        for text_part in text_parts:
            words = text_part.split(' ')
            for i, word in enumerate(words):
                pretty_text += word + ' '
                if (i + 1) % words_per_line == 0 and i != len(words) - 1:
                    pretty_text += '\n'
            pretty_text += '\n'
        
        return pretty_text
    
    
    @staticmethod
    def pretty_output_with_context(answer: str, context: list) -> str:
        return_str = f"{answer}\n\nKontext:\n"
        for doc in context:
            file_path = doc.metadata["source"]
            formatted_path = file_path.replace("\\", "/").replace("Data", "Source")
            encoded_path = urllib.parse.quote(formatted_path)
            file_url = f"file:///{encoded_path}"
            return_str += f"- [{os.path.basename(file_path)}]({file_url}) - Seite {doc.metadata['page_number']}\n"
        
        return return_str

class PartyRetriever(BaseRetriever):

    vectorstore: VectorStore
    embeddings: Embeddings

    def __init__(self, vectorstore: VectorStore, embeddings: Embeddings):
        super().__init__(vectorstore=vectorstore, embeddings=embeddings)
        self.embeddings = embeddings
        self.vectorstore = vectorstore

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        results = []
        query_embedding = self.embeddings.embed_query(query)
        for party in docs.keys():
            results += self.vectorstore.similarity_search_by_vector(query_embedding, k=3, filter={'party': party})

        return results


class Generator:

    def __init__(self, client):
        self.vectorstore = Chroma(
            collection_name=f"BTW2025",
            client=getClient(),
            create_collection_if_not_exists=False
        )

        self.llm = ChatOpenAI(
            api_key=OPENAI_API_KEY,
            model="gpt-4o-mini",
            temperature=0.0,
        )

        self.prompt =  ChatPromptTemplate([
        ("system", """Du bist ein Experte für politische Fragen zur Bundestagswahl und beantwortest die Fragen der Benutzer auf Basis des bereitgestellten Kontext. 
            Der Kontext besteht aus eine Aufstellung der Aussagen einzelner Parteien zu der Fragestellung des Benutzers.

            - Wenn die Frage anhand des Kontext beantwortet werden kann, gib in Deiner Antwort jeweils an, zu welcher Partei eine Aussage gehört.
            - Wenn es Aussagen mehrerer Parteien gibt, stelle die Aussagen der Parteien gegenüber und verdeutliche die Unterschieder der Parteien.
            - Wenn die Frage im Kontext nicht eindeutig beantwortet werden kann oder keine ausreichenden Informationen vorliegen, gib an, dass du die Frage nicht beantworten kannst.
            - Achte besonders darauf, dass du keine Informationen hinzufügst, die nicht im Kontext enthalten sind.
            - Gib am Ende Zitate aus den Aussagen der Parteien an, die Deine Zusammenfassung nachvollziehbar machen.

            Wenn in der Frage nach der Position einer bestimmten Partei gefragt wird, gehe in der Antwort auf diese Partei ein.
            Wenn in der Frage keine Partei explizit erwähnt wird, erstelle eine Übersicht der Positionen der folgenden Parteien:
            - CDU
            - SPD
            - Grüne
            - AfD
            - FDP
            - BSW
            - Linke

            Am Ende deiner Antwort weise bitte darauf hin, dass du ein ChatBot bist und die Antwort unbedingt von einer qualifizierten Person überprüft werden sollte.

            <kontext>
            {context}
            </kontext>"""),
                ("human", "Frage: {input}")
            ])

        self.embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=OPENAI_API_KEY)
        self.retriever = PartyRetriever(self.vectorstore, self.embeddings)
        self.chain = self.getChain()
        self.context = []
    


    def getChain(self):
        return create_retrieval_chain(
            retriever=self.retriever,
            combine_docs_chain=create_stuff_documents_chain(
                llm=self.llm,
                prompt=self.prompt,
                document_prompt=PromptTemplate.from_template("{party}: {page_content}")
            )
        )

    def invoke(self, query: str):
      
        res = self.chain.invoke({"input": query})
        self.context = res["context"]
        full_answer = PrettyOutput.pretty_output_with_context(res["answer"], res["context"])
        
        for i in range(len(full_answer)):
            yield full_answer[i:i+1]


@st.cache_resource
def getGenerator():
    return Generator(getClient())

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
        st.error("An Error Occured while clearing the Chat History", icon=ICON_ERROR)



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
            st.session_state.messages_history.append({"role": "user", "content": query})
            st.chat_message("user").write(query)
            
            generator = getGenerator()
            response_generator = generator.invoke(query)
            response_str = st.chat_message("ai").write_stream(response_generator)
            
            st.session_state.messages_history.append({"role": "ai", "content": response_str})
        except Exception as e:
            st.error(f"An Error Occured: {e}. Clear cache and try again.", icon=ICON_ERROR)


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
