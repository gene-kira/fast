import json
import os
import subprocess
from typing import List, Tuple
import logging

from qwen_agent.tools.base import register_tool
from qwen_agent.tools.doc_parser import Record
from qwen_agent.tools.search_tools.base_search import BaseSearch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def install_and_import(module_name: str, package_name: str = None):
    """
    Attempts to import a module and installs it if the import fails.
    """
    try:
        return __import__(module_name)
    except ModuleNotFoundError as e:
        logger.info(f"Module {module_name} not found. Installing {package_name or module_name}.")
        subprocess.check_call(['pip', 'install', package_name or module_name])
        return __import__(module_name)

@register_tool('vector_search')
class VectorSearch(BaseSearch):
    def __init__(self, embedding_model: str = 'text-embedding-v1', faiss_index_path: str = None):
        self.embedding_model = embedding_model
        self.faiss_index_path = faiss_index_path

        # Dynamically install and import required libraries
        self.Document = install_and_import('langchain.schema').Document
        self.DashScopeEmbeddings = install_and_import('langchain_community.embeddings', 'langchain-community').DashScopeEmbeddings
        self.FAISS = install_and_import('langchain_community.vectorstores', 'langchain-community').FAISS

    def sort_by_scores(self, query: str, docs: List[Record], **kwargs) -> List[Tuple[str, int, float]]:
        # Extract raw query
        try:
            query_json = json.loads(query)
            if 'text' in query_json:
                query = query_json['text']
        except json.decoder.JSONDecodeError:
            logger.warning("Query is not a valid JSON. Using the original query string.")

        # Plain all chunks from all docs
        all_chunks = []
        for doc in docs:
            for chk in doc.raw:
                if not chk.content or not chk.metadata.get('source') or not chk.metadata.get('chunk_id'):
                    logger.warning(f"Skipping chunk due to missing content or metadata: {chk}")
                    continue
                all_chunks.append(self.Document(page_content=chk.content[:2000], metadata=chk.metadata))

        if not all_chunks:
            logger.info("No valid chunks found to index.")
            return []

        # Initialize embedding model and FAISS index
        embeddings = self.DashScopeEmbeddings(model=self.embedding_model, dashscope_api_key=os.getenv('DASHSCOPE_API_KEY', ''))
        
        if self.faiss_index_path and os.path.exists(self.faiss_index_path):
            logger.info(f"Loading FAISS index from {self.faiss_index_path}")
            db = self.FAISS.load_local(self.faiss_index_path, embeddings)
        else:
            logger.info("Creating a new FAISS index")
            db = self.FAISS.from_documents(all_chunks, embeddings)
            if self.faiss_index_path:
                logger.info(f"Saving FAISS index to {self.faiss_index_path}")
                db.save_local(self.faiss_index_path)

        chunk_and_score = db.similarity_search_with_score(query, k=len(all_chunks))

        return [(chk.metadata['source'], chk.metadata['chunk_id'], score) for chk, score in chunk_and_score]

# Example usage
if __name__ == "__main__":
    # Assuming you have a list of Record objects named `documents`
    query = "How does vector search work?"
    vector_search = VectorSearch(embedding_model='text-embedding-v1', faiss_index_path='./faiss_index')
    results = vector_search.sort_by_scores(query, documents)
    for source, chunk_id, score in results:
        print(f"Source: {source}, Chunk ID: {chunk_id}, Score: {score}")
