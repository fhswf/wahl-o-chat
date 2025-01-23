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
        "BSW": "BSW_Wahlprogramm_2025.pdf",
        "Grüne": "Grüne_BTW2025.pdf",
        "CDU": "CDU_BTW2025.pdf",
        "AfD": "AfD_Leitantrag-Bundestagswahlprogramm-2025.pdf",
        "Linke": "btw_2025_wahlprogramm_die_linke.pdf",
        "SPD": "BTW_2025_SPD_Regierungsprogramm.pdf",
        "FDP": "fdp-wahlprogramm_2025.pdf",
        "Volt": "volt-programm-bundestagswahl-2025.pdf"
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
            results += self.vectorstore.similarity_search_by_vector(
                query_embedding, k=3, filter={'party': party})

        return results
