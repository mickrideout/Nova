import logging
from agent.base_agent import BaseAgent
from vo.idea_data import SearchPlan
from utils_tool import extract_json_between_markers

# Initialize logger
logger = logging.getLogger(__name__)

class IdeaSearchPlanAgent(BaseAgent):
    """Agent for creating search plans to explore indirect knowledge across various fields."""

    def __init__(self, args):
        """Initialize the IdeaSearchPlanAgent with necessary arguments and configurations."""
        BaseAgent.__init__(self, args)
        self.args = args
        self.system_msg = """"""
        self.task_name = "IdeaSearchPlanAgent"
        self.plan_json_format = """In <JSON>, provide the search plan list in JSON format, every plan with the following fields:
            - "title": Indicate the field and direction in which you want to search for literature
            - "thinking": The thought process behind proposing the plan.
            - "keywords": keywords that can help search in google scholar then you can find related paper.
            - "rationale": The reason for generating this plan, explain why this should help you gather comprehensive information and new knowledge related to the input idea and target paper.
            """
        with open("prompts/idea_plan_example.txt", "r") as f:
            self.few_shot_example = f.read().strip()
        # self.few_shot_example = """
        #     # Example
        #     here is an example, you should follow the plan example formatted (but don't borrow the plan themselves)
        #     Research idea:
        #         idea: Exploring the long-term impact of LLM-generated research ideas on scientific progress and innovation。
        #         keywords: long-term impact, scientific progress, innovation, LLM-generated research ideas
        #         thinking: Understanding the long-term effects can guide future development and application of LLM in research.
        #     Search plan output:
        #     In order to explore the long-term impact of LLM-generated research ideas on scientific progress and innovation, we need to collect relevant literature from multiple disciplines and fields. Here are my suggested query areas, thought processes, and keywords for each area:
        #     ```json
        #     [
        #         {
        #             "title": "How LLM or Chatgpt Generates Research Ideas",
        #             "thinking": "First, we need to understand the development of LLM technology itself, especially how LLM generates research ideas. This involves machine learning models, algorithms, and their applications in generating new ideas.",
        #             "keywords": [
        #                 "LLM research idea generation",
        #                 "LLM Inspiration generation",
        #                 "LLM creativity",
        #                 "OpenLLM",
        #                 "GPT4",
        #                 "Ideate"
        #             ]
        #         },
        #         {
        #             "title": "Philosophy of Science and History of Science",
        #             "thinking": "Studying the impact of LLM on scientific progress requires examining the nature and process of scientific development from a philosophical and historical perspective. This helps us understand how LLM may change the way of scientific exploration.",
        #             "keywords": [
        #                 "philosophy of science",
        #                 "history of scientific innovation",
        #                 "scientific methodology",
        #                 "LLM in scientific history"
        #             ]
        #         },
        #         {
        #             "title": "Sociology and sociology of science",
        #             "thinking": "The sociological perspective can help us understand how LLM-generated research ideas affect the structure and dynamics of the scientific community, and how these changes affect scientific progress.",
        #             "keywords": [
        #                 "sociology of science",
        #                 "scientific community",
        #                 "LLM impact on scientific sociology",
        #                 "social dynamics of innovation"
        #             ]
        #         },
        #         {
        #             "title": "Policy research and science and technology policy",
        #             "thinking": "Policy research can provide idea on how to guide the application of LLM in scientific research through policies, and how these policies affect scientific progress and innovation.",
        #             "keywords": [
        #                 "science policy",
        #                 "LLM policy",
        #                 "long-term policy impact",
        #                 "innovation policy"
        #             ]
        #         }
        #     ]
        #     ```
        #     By cross-referencing these fields, we can obtain a comprehensive perspective that includes not only technical-level analysis, but also social, historical, and policy-level considerations. Such a multi-dimensional analysis will help us to have a deeper understanding of the long-term impact of LLM-generated research ideas on scientific progress and innovation.
        #     """

    def _gen_prompt(self, idea_info, target_paper_title=""):
        """Generate the prompt for the LLM based on the given research idea and target paper title."""
        prompt = f"""
            # Role
            You are an expert researcher in AI. You can think out of the box, develop a detailed paper search plan for a given research idea base on target paper.
            # Skill
            Here is a research idea for you, please analyze the sequence of different fields which you should search for relevant papers.
            This way, you can gather comprehensive information and new knowledge, further expanding your research perspective and for finding new ideas.
            # Example
            You can follow these examples to get a sense of how the plan should be formatted (but don't borrow the plan themselves):
            {self.few_shot_example}
            # Format
            {self.plan_json_format}
            # Requirement
            1. The search_plan needs to be developed around the given research idea
            2. The plan needs to focus on how to optimize Target paper
            3. Please provide the thought process and search keywords.
            4. Please output the Thinking process.
            5. please thinking step by step
            # Input:
            Research idea: {idea_info}
            Target paper title:{target_paper_title}
            # Output:
            Thinking: <Please output the Thinking process here.>
            Search plan output:
            <JSON>
            """
        logger.info(f"Generated prompt for idea: {idea_info} and target paper: {target_paper_title}")
        return prompt

    def run(self, idea: str, target_paper_title="") -> SearchPlan:
        """Execute the search plan generation process."""
        logger.info(f"Starting search plan generation for idea: {idea}")
        prompt = self._gen_prompt(idea, target_paper_title)
        retry_times = 3
        for i in range(retry_times):
            try:
                llm_response = self._call_llm(
                    self.system_msg,
                    prompt,
                    parse_data_flag=False
                )
                search_plan = self._parse_data(
                    llm_response=llm_response,
                    sys="",
                    prompt=prompt,
                    llm_result=llm_response
                )
                assert search_plan.success == True, f"plan_agent | {llm_response}"
                logger.info(f"Search plan generated successfully for idea: {idea}")
                return search_plan
            except Exception as e:
                logger.info(f"Search plan generated failed for idea: {idea}, please check!, e:{e}")
                return None

    def _parse_data(self, llm_response, sys=None, prompt=None, llm_result=None):
        """Parse the LLM response to extract the search plan."""
        logger.info("Parsing LLM response to extract search plan")
        plan_list = extract_json_between_markers(llm_response)
        if len(plan_list) > 0:
            logger.info("Search plan extracted successfully")
            return SearchPlan(plan_list=plan_list, prompt=prompt, sys=sys, llm_result=llm_result, success=True)
        logger.warning("No valid search plan found in LLM response")
        return SearchPlan(plan_list=[], prompt=prompt, sys=sys, llm_result=llm_result, success=False)