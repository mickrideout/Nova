import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify
from agent.research_trending_agent import PopularAgent
import traceback
import logging
from functools import lru_cache
import concurrent.futures
from args_tool import get_args
from utils_tool import load_pickle_from_file, convert_float32, init_logging
from src.academic_search_engine import PaperSearchEngine

# Initialize logging
logger = logging.getLogger()
init_logging(log_dir="log", log_filename="research_trending_and_rag_server.log")

# Parse command line arguments
args = get_args()

# Initialize PopularAgent
popularAgent = PopularAgent(args)
logger.info("Initialized PopularAgent successfully.")

# Initialize Flask app
app = Flask(__name__)

# Load FAISS index and related documents
FAISS_INDEX_DIR = args.FAISS_INDEX_DIR
EMBEDDING_MODEL_DIR = args.EMBEDDING_MODEL_DIR
RERANK_MODEL_DIR = args.RERANK_MODEL_DIR

# Load document list
doc_file = f"{FAISS_INDEX_DIR}/doc_list.pkl"
DOCUMENTS = load_pickle_from_file(doc_file)
logger.info(f"Loaded {len(DOCUMENTS)} documents from {doc_file}")

# Load paper ID list
paper_id_file = f"{FAISS_INDEX_DIR}/paper_id_list.pkl"
PAPER_ID_LIST = load_pickle_from_file(paper_id_file)
logger.info(f"Loaded {len(PAPER_ID_LIST)} paper IDs from {paper_id_file}")

# Load FAISS index
FAISS_INDEX_FILE = f"{FAISS_INDEX_DIR}/index.faiss"
paperRagFaiss = PaperSearchEngine(
    args=None,
    embedding_model_name=EMBEDDING_MODEL_DIR,
    rerank_model_file=RERANK_MODEL_DIR
)
paperRagFaiss.load_index(FAISS_INDEX_FILE)
logger.info(f"Loaded FAISS index from {FAISS_INDEX_FILE}")

def get_research_trending(paper_str):
    """
    Get popular research directions from a given paper string.
    :param paper_str: String containing paper titles and abstracts.
    :return: List of popular research directions or None if not found.
    """
    llm_result = popularAgent.run(paper_str)
    data = llm_result.to_dict()
    if data and "popular_research_directions" in data:
        return data.get("popular_research_directions")
    return None

def validate_input(data):
    """
    Validate the input data to ensure it includes 'paper_list' and each paper has 'title' and 'abstract_info'.
    :param data: Input data to validate.
    :return: Tuple of (is_valid, error_message).
    """
    if not data or 'paper_list' not in data:
        return False, 'Invalid input: paper_list is required'

    if not data or 'request_id' not in data:
        return False, 'Invalid input: request_id is required'

    paper_list = data['paper_list']

    if not isinstance(paper_list, list) or not all(isinstance(paper, dict) for paper in paper_list):
        return False, 'Invalid input: paper_list must be a list of dictionaries'

    for paper in paper_list:
        if 'title' not in paper or 'abstract_info' not in paper:
            return False, 'Invalid input: each paper must include title and abstract_info'

    return True, ''

@lru_cache(maxsize=128)
def search_and_rank(search_query, rank_doc, topk=10):
    """
    Search and rank documents based on the search query and rank document.
    :param search_query: Search query string.
    :param rank_doc: Document used for ranking.
    :param topk: Number of top documents to return.
    :return: List of topk documents with scores [[doc, score], ...].
    """
    ids, distances = paperRagFaiss.search(search_query, topk=topk)
    topk_search_result = [DOCUMENTS[i] for i in ids]
    sorted_document_with_score = paperRagFaiss.rerank(rank_doc, topk_search_result)
    return sorted_document_with_score

@lru_cache(maxsize=128)
def search_only(search_query, rank_doc, topk=10):
    """
    Search documents based on the search query without ranking.
    :param search_query: Search query string.
    :param rank_doc: Document used for ranking.
    :param topk: Number of top documents to return.
    :return: List of topk documents with scores [[doc, score], ...].
    """
    ids, distances = paperRagFaiss.search(search_query, topk=topk)
    distances = [convert_float32(_) for _ in distances]
    topk_search_result = [DOCUMENTS[i] for i in ids]
    results = []
    for result, dis in zip(topk_search_result, distances):
        results.append([result, dis])
    return results

@lru_cache(maxsize=128)
def search_and_rank_v2(search_query, topk=10):
    """
    Search and rank documents based on the search query.
    :param search_query: Search query string.
    :param topk: Number of top documents to return.
    :return: List of topk documents with detailed information [{id, title, abstract, score}...].
    """
    ids, distances = paperRagFaiss.search(search_query, topk=topk)
    distances = [convert_float32(_) for _ in distances]
    logger.info(f"search_and_rank_v2 | distances:{distances}")
    topk_search_result = [DOCUMENTS[i] for i in ids]
    topk_paper_id_result = [PAPER_ID_LIST[i] for i in ids]
    topk_search_result_id_lookup = {DOCUMENTS[i]: i for i in ids}
    paper_list = []
    for doc, arxiv_id, score in zip(topk_search_result, topk_paper_id_result, distances):
        title, abstract = doc.split(', Abstract:')
        paper = {
            'arxiv_id': arxiv_id,
            'title': title.split('Title:')[1].strip(),
            'abstract': abstract.strip(),
            'score': score,
        }
        paper_list.append(paper)
    return paper_list

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/paper_search', methods=['POST'])
def search():
    """
    Flask endpoint to handle search requests with ranking.
    :return: JSON response containing search results.
    """
    data = request.json
    logger.info(f"paper_search | request_data:{data}")
    search_query = data.get('search_query', '')
    rank_doc = data.get('rank_doc', '')
    topk = int(data.get('topk', 10))
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(search_and_rank, search_query, rank_doc, topk)
        result = future.result()
    return jsonify(result)

@app.route('/paper_search_v2', methods=['POST'])
def search_v2():
    """
    Flask endpoint to handle search requests with detailed paper information.
    :return: JSON response containing search results.
    """
    data = request.json
    query = data.get('query', '')
    topk = int(data.get('topk', 10))
    logger.info(f"paper_search_v2 | query:{query}, k:{topk}")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(search_and_rank_v2, query, topk)
        result = future.result()
    return jsonify(result)

@app.route('/paper_search_only', methods=['POST'])
def search_v3():
    """
    Flask endpoint to handle search requests without ranking.
    :return: JSON response containing search results.
    """
    data = request.json
    logger.info(f"paper_search_only | request_data:{data}")
    search_query = data.get('search_query', '')
    topk = int(data.get('topk', 10))
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(search_only, search_query, topk)
        result = future.result()
    return jsonify(result)

@app.route('/compute_embedding', methods=['POST'])
def compute_embedding():
    """
    Flask endpoint to compute embeddings for given texts.
    :return: JSON response containing embeddings.
    """
    data = request.json
    texts = data['texts']
    embeddings = paperRagFaiss.embedding_model.encode(texts).tolist()
    return jsonify(embeddings)

@app.route('/nova/ai/get_research_trending_from_paper_list', methods=['POST'])
def get_research_trending_from_paper_list():
    """
    Flask endpoint to get research trends from a list of papers.
    :return: JSON response containing research trends.
    """
    try:
        data = request.get_json()
        logger.info(f"get_research_trending_from_paper_list | Received data: {data}")
        is_valid, error_msg = validate_input(data)

        if not is_valid:
            logger.info(f"Validation failed: {error_msg}")
            return jsonify({
                'result': '',
                'msg': error_msg,
                'code': 400
            }), 400

        paper_list = data['paper_list']
        paper_str = " ".join(paper['title'] + paper['abstract_info'] for paper in paper_list)
        trending_summary = get_research_trending(paper_str)
        assert trending_summary is not None

        logger.info(f"Trending summary: {trending_summary}")
        return jsonify({
            'result': trending_summary,
            'msg': 'Success',
            'code': 200
        }), 200

    except Exception as e:
        logger.error(f"Exception occurred: {traceback.format_exc()}")
        return jsonify({
            'result': '',
            'msg': 'Internal server error',
            'code': 500
        }), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True, port=9918)