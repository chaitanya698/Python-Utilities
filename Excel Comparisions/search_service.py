import logging
from typing import List, Dict, Any, Optional

from langchain.retrievers import TFIDFRetriever
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

logger = logging.getLogger('SearchService_logger')

class SearchService:
    def __init__(self, documents: Optional[List[Document]] = None):
        self.embeddings = OpenAIEmbeddings()
        self.llm = OpenAI(temperature=0)
        
        # Initialize vector store if documents are provided
        if documents:
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            )
        else:
            # Default to TFIDF if no documents provided
            self.retriever = TFIDFRetriever()
            
        # Create QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True
        )
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the search index."""
        try:
            if hasattr(self, 'vector_store'):
                self.vector_store.add_documents(documents)
            else:
                self.vector_store = FAISS.from_documents(documents, self.embeddings)
                self.retriever = self.vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 5}
                )
                
                # Update QA chain with new retriever
                self.qa_chain = RetrievalQA.from_chain_type(
                    llm=self.llm,
                    chain_type="stuff",
                    retriever=self.retriever,
                    return_source_documents=True
                )
                
            logger.info(f"Added {len(documents)} documents to search index")
        except Exception as e:
            logger.error(f"Error adding documents to search index: {e}")
            raise
    
    def search(self, query: str, k: int = 5) -> List[Document]:
        """Search for documents related to the query."""
        try:
            results = self.retriever.get_relevant_documents(query)
            return results[:k]
        except Exception as e:
            logger.error(f"Error searching for documents: {e}")
            return []
    
    def query(self, question: str) -> Dict[str, Any]:
        """Query the QA system with a question."""
        try:
            result = self.qa_chain({"query": question})
            return {
                "answer": result["result"],
                "source_documents": result.get("source_documents", [])
            }
        except Exception as e:
            logger.error(f"Error querying QA system: {e}")
            return {
                "answer": f"Error: {str(e)}",
                "source_documents": []
            }
    
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """Perform similarity search on the vector store."""
        try:
            if hasattr(self, 'vector_store'):
                return self.vector_store.similarity_search(query, k=k)
            else:
                logger.warning("Vector store not initialized for similarity search")
                return []
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            return []