from typing import List
from collections.abc import Mapping

from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.vectorstores.base import VectorStore
from langchain_core.embeddings.embeddings import Embeddings
from langchain_core.documents.base import Document





class PartyRetriever(BaseRetriever):
    """
    A retriever that retrieves documents from the party programs"""
    
    docs: Mapping[str, str] = {
        "BSW": "BSW_Parteiprogramm.pdf",
        "Grüne": "Grüne_BTW2025.pdf",
        "CDU": "CDU_BTW2025.pdf",
        "AfD": "Programm_AfD_Online_.pdf",
        "Linke": "DIE_LINKE_Wahlprogramm_zur_Bundestagswahl_2021.pdf",
        "SPD": "SPD-Zukunftsprogramm.pdf",
        "FDP": "FDP_Programm_Bundestagswahl2021_1.pdf"
    }
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
        for party in self.docs:
            results += self.vectorstore.similarity_search_by_vector(query_embedding, k=3, filter={'party': party})

        return results