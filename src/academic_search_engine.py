import logging
from sentence_transformers import SentenceTransformer, CrossEncoder
import pickle
import faiss
from utils_tool import convert_float32

logger = logging.getLogger()

class PaperSearchEngine:

    def __init__(self, args=None, embedding_model_name="rag/all-MiniLM-L6-v2", rerank_model_file="rag/ms-marco-MiniLM-L-6-v2"):
        # Initialize the embedding model and the reranking model
        logger.info(f"PaperRagFaiss | load embedding model from:{embedding_model_name}")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        logger.info(f"PaperRagFaiss | load rerank model from:{rerank_model_file}")
        self.reranker = CrossEncoder(rerank_model_file)
        self.index = None
        self.doc_ids = None

    def get_embeddings(self, doc_list):
        # Compute document embeddings and save them to a file
        embeddings = self.embedding_model.encode(doc_list, show_progress_bar=True)
        return embeddings


    def save_embeddings(self, doc_list, embedding_file):
        # Calculate document embedding and save to file
        embeddings = self.embedding_model.encode(doc_list, show_progress_bar=True)
        with open(embedding_file, 'wb') as f:
            pickle.dump(embeddings, f)

    def load_embeddings(self, embedding_file):
        with open(embedding_file, 'rb') as f:
            embeddings = pickle.load(f)
        return embeddings

    def build_index(self, embeddings):
        # build FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # 使用余弦相似度
        self.index.add(embeddings)

    def save_index(self, index_file):
        # Persistent FAISS Index
        faiss.write_index(self.index, index_file)

    def load_index(self, index_file):
        # Load the persistent FAISS index
        self.index = faiss.read_index(index_file)

    def search(self, query, topk=5):
        logger.info(f"PaperRagFaiss | query:{query}, topk:{topk}")
        query_embedding = self.embedding_model.encode([query])
        distances, indices = self.index.search(query_embedding, topk)
        return indices[0], distances[0]



    def rerank_related_papers(self, query, related_papers):
        # Create pairs of (query, document)
        pairs = [[query, paper['content']] for paper in related_papers]
        # Get scores for each pair
        scores = self.reranker.predict(pairs)
        scores = [convert_float32(_) for _ in scores]
        results = []
        for paper, score in zip(related_papers, scores):
            paper['score'] = score
            results.append(paper)
        return {
            "query": query,
            "related_papers": results
        }

    def rerank(self, query, documents):
        # Create pairs of (query, document)
        pairs = [[query, doc] for doc in documents]
        # Get scores for each pair
        scores = self.reranker.predict(pairs)
        scores = [convert_float32(_) for _ in scores]
        # Sort documents by score in descending order
        sorted_documents = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        return sorted_documents
