import os.path
import pickle
import time
from collections import deque
from typing import List
import logging

logger = logging.getLogger()

from agent.seed_idea_agent import SeedIdeaAgent
from agent.new_idea_iteration_agent import IdeaIterationAgent
from agent.plan_agent import IdeaSearchPlanAgent
from src.academic_search_tool import search_papers_according_to_query
from utils_tool import load_json_from_file, save_json_data_to_file
from vo.idea_data import IdeaResult, load_SeedIdeaResult_from_dict
from vo.paper_data import Paper

class IdeaRelatedPaperPlanThenRetrievalAgent:
    """Find complete information related to the idea
    1. Create a search plan
    2. Execute the search plan sequentially, analyze the search results, and iterate to optimize the search results
    TODO: Iterate and optimize during the search process, not implemented yet as it consumes a lot of tokens
    """
    def __init__(self, args=None, use_llm_check_rag_result=False):
        self.args = args
        self.plan_agent = IdeaSearchPlanAgent(args)
        self.use_llm_check_rag_result = use_llm_check_rag_result

    def run(self, idea: str, top_k: int, target_paper_title="") -> dict:
        """Retrieve related idea information
        Args:
            idea (str): Analyzed potentially useful idea
            top_k (int): Number of materials to query for each topic
        Returns:
            return_info (dict):
                idea: Current idea information
                prompt: Search plan prompt
                llm_result: Search plan llm result
                search_plan_result_list: All search plans and results
                    search_info: Key for searching articles
                    top1_document: Top 1 result
                    sorted_document_with_score: All search results
                    obs_and_thinking_list: Observations and thinking from search results, used to analyze if search terms need updating
        """
        return_info = {
            "idea": idea
        }
        # 1. Create a search plan
        search_plan_vo = self.plan_agent.run(idea, target_paper_title=target_paper_title)
        assert search_plan_vo.success == True
        search_plan_list = search_plan_vo.get_search_plan()
        return_info["prompt"] = search_plan_vo.get_prompt()
        return_info["llm_result"] = search_plan_vo.get_llm_response()
        results = []
        # 2. Execute specific search plans
        search_plan_result_list = []
        plan_detail = ""
        retrieval_detail = ""
        exist_top1_document = set()
        for search_i, search_plan in enumerate(search_plan_list):
            topk_search_result, obs, thinking_result = [], "", ""
            # Search up to 3 times
            search_info = "Keywords:" + str(search_plan['keywords']) + ", Title:" + search_plan['title'] + ", Thinking:" + search_plan['thinking']
            plan_detail += f'{search_i}. {search_plan["thinking"]}\n'
            sorted_document_with_score = []
            for retry_time in range(self.args.MAX_RETRY_TIMES_FOR_ACADEMIC_SEARCH_ENGINE):
                try:
                    sorted_document_with_score = search_papers_according_to_query(
                        search_query=search_info,
                        topk=top_k,
                        ip_address=self.args.ACADEMIC_SEARCH_ENGINE_IP_ADDRESS
                    )
                    assert len(sorted_document_with_score) > 0, "plan search result sorted_document_with_score size is 0"
                    logger.info(f"search_papers_according_to_query success at {retry_time} times")
                    break
                except Exception as e:
                    logger.info(f"search_papers_according_to_query failed at {retry_time} times, msg:{e}")
                    time.sleep(retry_time)
            assert len(sorted_document_with_score) > 0, "plan search result sorted_document_with_score size is 0"
            # Select the top 1 document, but remove possible duplicates
            top1_document = ''
            for doc, score in sorted_document_with_score:
                if doc not in exist_top1_document:
                    top1_document = doc
                    exist_top1_document.add(doc)
                    break
            search_plan['search_info'] = search_info
            search_plan['sorted_document_with_score'] = sorted_document_with_score
            search_plan['top1_document'] = top1_document
            doc = top1_document.replace("\n", "")
            retrieval_detail += f'{search_i}. {doc}\n'
            search_plan['obs_and_thinking_list'] = []
            logger.info(f"InsightRagAgent | run | search_info:{search_info}, top1_document:{top1_document}")
            results.append([search_plan, top1_document, "", ""])
            search_plan_result_list.append(search_plan)
            # break
        return_info['search_plan_result_list'] = search_plan_result_list
        return_info['plan_detail'] = plan_detail
        return_info['retrieval_detail'] = retrieval_detail
        return return_info