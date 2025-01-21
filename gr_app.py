import gradio as gr
from gradio.data_classes import _StaticFiles
from gradio.components import State
import urllib.parse

import asyncio
import chromadb
import os
import time

from functools import reduce
from collections import defaultdict
from collections.abc import Awaitable
from aiostream import stream
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
from langchain.retrievers import MergerRetriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_chroma.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from Callback import GradioCallbackHandler

from Generator import Generator

load_dotenv(find_dotenv())

__version__ = "0.9.0"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DATABASE_PATH = "./chroma/"
EMBEDDING_MODEL = "text-embedding-ada-002"


ICON_DELETE = ":material/delete:"
ICON_CHECK = ":material/check:"
ICON_ERROR = ":material/error:"
ICON_WARNING = ":material/warning:"
ICON_INFO = ":material/info:"
ICON_RESTART_ALT = ":material/restart_alt:"


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

examples = [
    "Was sagen die Parteien zum Thema KI?",
    "Wie sind die Positionen der Parteien in der Hochschulpolitik?",
    "Wie wollen die Parteien Investitionen fördern?"
]



def user(user_message, history: list):
    return "", history + [{"role": "user", "content": user_message}], ""

def formatLink(doc):
    return f"""
- <a href="gradio_api/file/files/{doc.metadata['filename']}#page={doc.metadata['page_number']}" target="_blank">{"".join(doc.metadata['filename'].split('.')[:-1])}, Seite {doc.metadata['page_number']}</a>"""

def formatParty(record):
    (party, docs) = record
    text = f"### {party}\n\n"
    for d in docs:
        text += formatLink(d)
    return text

#category element_id file_directory filename filetype languages last_modified orig_elements page_number party source
# [SPD Zukunftsprogramm](gradio_api/file/files/SPD-Zukunftsprogramm.pdf)

def formatContext(context):
    header = """## Quellen\n\n"""

    by_party = defaultdict(list)
    for d in context:
        by_party[d.metadata['party']].append(d)

    return header + "\n\n".join(map(formatParty, by_party.items()))


async def chat(message, history, progress=gr.Progress()):
    history.append(gr.ChatMessage(role="user", content=message))
    print("history: ", history)
    yield history, ""
    cb = GradioCallbackHandler(progress)
    call = generator.ainvoke({"input": message}, {"callbacks": [cb]})

    tasks = []
    async for x in call:
        task = asyncio.create_task(x)
        task.add_done_callback(cb.end_run)
        tasks.append(task)
        print(f"{x} - processing submitted")

    print("task: ", type(task), tasks)
    print("callback: ", type(cb), isinstance(cb, Awaitable))

    
    async for x in cb:
        r = x
        print("got: ", type(r), str(r)[:80])
        if isinstance(r, str):
            #history.append(gr.ChatMessage(role="assistant", metadata={"title": "thinking ..."}, content=r))
            yield history, ""

        elif "answer" in r:
            history.append(gr.ChatMessage(role="assistant", content=r["answer"]))
            yield history, ""

    result = await asyncio.gather(*tasks)
    for r in result:
        history.append(gr.ChatMessage(role="assistant", content=r["answer"]))
        yield history, formatContext(r["context"])

gr.set_static_paths(paths=["files/"])
    
history = [ gr.ChatMessage(role="assistant", 
                           content="""Guten Tag, ich habe die **Wahlprogramme** der Parteien zur **Bundestagswahl** am **23. Februar 2025** gelesen und beantworte gerne Deine Fragen dazu!
                                      Meinen Quellcode findest Du übrigens auf [GitHub](https://github.com/fhswf/wahl-o-chat).
                                   """)
          ]
with gr.Blocks(title="Wahl-o-Chat", fill_height=True) as demo:
    with gr.Tab("Chat"):
        chatbot = gr.Chatbot(value=history, type="messages", label="Wahl-o-Chat", min_height=400, height=None)
    with gr.Tab("Quellen"):
        references = gr.Markdown(""" """)
    message = gr.Textbox(submit_btn=True, show_label=False, placeholder="Gib hier Deine Frage ein")
    saved_message = gr.State()
    
    message.submit(lambda x: [ "", x ], message, [message, saved_message]).then(chat, [saved_message, chatbot], [chatbot, references])

print(_StaticFiles.all_paths)
if __name__ == "__main__":
    demo.launch(pwa=True)
