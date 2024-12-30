from PartyRetriever import PartyRetriever
from PrettyOutput import PrettyOutput
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.runnables.config import RunnableConfig

class Generator:
    """
    A generator that generates answers to questions based on the context of the question"""

    def __init__(self, vectorstore, embeddings, llm):
        self.vectorstore = vectorstore

        self.llm = llm

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

        self.embeddings = embeddings
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


    def invoke(self, input: dict, config: RunnableConfig):
        res = self.chain.invoke(input, config)
        self.context = res["context"]
        return res