import requests
import json
import logging

logger = logging.getLogger()

def search_papers_according_to_query(search_query, rank_doc=None, topk=10, ip_address="http://0.0.0.0:9918/"):
    """
    Args:
        search_query (str): The search query.
        rank_doc (str): The document to rank against.
        topk (int): The number of top results to return.
    Returns:
        list: A list of topk search results with scores.
    """
    url = f"{ip_address}/paper_search_only"

    payload = {
        "search_query": search_query,
        "rank_doc": search_query,
        "topk": topk
    }

    headers = {"content-type": "application/json"}

    response = requests.request("POST", url, json=payload, headers=headers)

    return json.loads(response.text)

def search_paper(query, k=5, url='http://0.0.0.0:9918/paper_search_v2'):
    logger.info(f"search_paper | query:{query}, k:{k}, url:{url}")
    payload = {
        "query": query,
        "topk": k
    }
    headers = {"content-type": "application/json"}
    response = requests.request("POST", url, json=payload, headers=headers)
    # print("text:", response.text)
    return json.loads(response.text)


if __name__ == "__main__":

    from args_tool import get_args
    args = get_args()
    
    results = search_paper("ResearchAgent", k=5, url=f"{args.ACADEMIC_SEARCH_ENGINE_IP_ADDRESS}/paper_search_v2")
    for paper in results:
        print(paper)


    results = search_papers_according_to_query(search_query="ResearchAgent", ip_address=args.ACADEMIC_SEARCH_ENGINE_IP_ADDRESS)
    for paper in results:
        print(paper)


