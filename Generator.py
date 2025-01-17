from PartyRetriever import PartyRetriever
from PrettyOutput import PrettyOutput
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.runnables.config import RunnableConfig
from langchain_core.runnables import RunnableSequence, Runnable, RunnablePassthrough
from langchain_core.output_parsers.json import JsonOutputParser
from langchain_core.output_parsers.openai_tools import PydanticToolsParser

from typing import Any, Dict, Union
from pydantic import BaseModel, Field


class ResponseFormatter(BaseModel):
    """Always use this tool to structure your response to the user."""
    id: str = Field(description="ID of the document")
    score: int = Field(description="The relevance of the document")


class ContextCleanup(Runnable[Dict[str, Any], Dict[str, Any]]):
    # TODO: Sortiere Dokumente nach Relevanz
    def invoke(self, inputs, config, **kwargs):
        ranking = {x.id: x.score for x in inputs['ranking']}
        old_context = inputs['context']
        context = []
        print("context cleanup", ranking)
        new_context = []
        for d in old_context:
            id = d.metadata['element_id']
            score = ranking[id] if id in ranking else 0
            print("context cleanup: ", id, score)
            if score > 0:
                context.append(d)
        return context


class Generator:
    """
    A generator that generates answers to questions based on the context of the question"""

    def __init__(self, vectorstore, embeddings, llm):
        self.vectorstore = vectorstore

        self.llm = llm

        self.prompt = ChatPromptTemplate([
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

            Achte darauf, dass Du *keine* konkrete Wahlempfehlung für eine bestimmte Partei aussprichst, sondern nur Informationen zu den Positionen der Parteien gibst.
            Am Ende deiner Antwort weise bitte darauf hin, dass du ein ChatBot bist und die Antwort anhand der Quellen überprüft werden sollte.

            <kontext>
            {context}
            </kontext>"""),
            ("human", "Frage: {input}")
        ])

        self.rankingPrompt = ChatPromptTemplate([
            ("system", """Du bist ein Experte für politische Fragen zur Bundestagswahl und beantwortest die Fragen der Benutzer auf Basis des bereitgestellten Kontext. 
            Der Kontext besteht aus eine Aufstellung der Aussagen einzelner Parteien zu der Fragestellung des Benutzers.

            In einem ersten Schritt besteht Deine Aufgabe darin, die Relevanz der Dokumente für die Beantwortung der Frage auf einer Skala von 0 (keine Relevanz)
            bis 100 (sehr hohe Relevanz) zu bewerten.
            Beachte dabei folgende Vorgaben:
            - Wenn in der Frage nach der Position einer bestimmten Partei gefragt wird, gebe den Stellungnahmen der anderen Parteien den Wert 0.
            - Wenn in der Frage keine Partei explizit erwähnt wird, beurteile die Relevanz rein inhaltlich für alle Parteien und vergebe nur Werte im Bereich 10 – 100, 
              damit keine Aussage eine Partei ganz herausgefiltert wird. 

            Formatiere Deine Ausgabe wie folgt als Dictionary:
            {{ "ID": <ID des Dokuments>, "score": <Relevanz als Zahl von 0 bis 100>, ... }}

            <kontext>
            {context}
            </kontext>"""),
            ("human", "Frage: {input}")
        ])

        self.embeddings = embeddings
        self.retriever = PartyRetriever(self.vectorstore, self.embeddings)
        self.rerank_chain = create_stuff_documents_chain(
            llm=self.llm.bind_tools([ResponseFormatter]),
            # llm=self.llm,
            output_parser=PydanticToolsParser(tools=[ResponseFormatter]),
            prompt=self.rankingPrompt,
            document_prompt=PromptTemplate.from_template(
                "Aussage der Partei {party} mit ID {element_id}: {page_content}")
        )
        self.combine_chain = create_stuff_documents_chain(
            llm=self.llm,
            prompt=self.prompt,
            document_prompt=PromptTemplate.from_template(
                "Aussage der Partei {party}: {page_content}")
        )

        retrieval_docs = (lambda x: x["input"]) | self.retriever

        self.chain = (
            RunnablePassthrough
            .assign(
                context=retrieval_docs.with_config(run_name="retrieve_documents", metadata={
                                                   "message": "Suche Informationen ..."}),
            )
            .assign(
                ranking=self.rerank_chain.with_config(metadata={"message": "Bewerte Informationen ..."}))
            .assign(context=ContextCleanup())
            .assign(answer=self.combine_chain.with_config(metadata={"message": "Erstelle Übersicht ..."}))
        ).with_config(run_name="retrieval_chain")

        self.context = []

    def invoke(self, input: dict, config: RunnableConfig):
        res = self.chain.invoke(input, config, verbose=True)
        self.context = res["context"]
        return res

    async def ainvoke(self, input: dict, config: RunnableConfig):
        res = self.chain.ainvoke(input, config, verbose=True)
        yield res
