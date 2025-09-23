import logging
from agent.base_agent import BaseAgent
from vo.trending_data import PopularResult

# Initialize logger
logger = logging.getLogger(__name__)

class PopularAgent(BaseAgent):
    """Agent to mine current popular research directions."""

    def __init__(self, args):
        """
        Initialize the PopularAgent.

        Args:
            args: Arguments required for initializing the BaseAgent.
        """
        BaseAgent.__init__(self, args)
        self.args = args
        self.system_msg = """"""
        self.task_name = "PopularAgent"
        logger.info(f"Initialized {self.task_name} with args: {self.args}")

    def _gen_search_term_prompt_by_paper(self, popular_paper_list):
        """
        Generate a prompt for analyzing popular research trends based on a list of recent papers.

        Args:
            popular_paper_list: List of recent popular papers.

        Returns:
            str: Generated prompt for the LLM.
        """
        prompt = f"""
            # Role
            You are an AI expert researcher. You can summarise the current hot research trends from the list of recent AI papers.
            # Skill
            You will analyze the research trending  based on the recent popular paper, provide us with the research trending report.
            # Requirements
            1. Provide a comprehensive analysis, including the hot research directions, the highlights of the technologies and methods, and discuss whether these technologies can be used in other fields.
            I will provide a list of recent popular paper list here:
            {popular_paper_list}
            Then, Please output the current research trending report here:
            """
        logger.debug(f"Generated prompt for popular paper list: {popular_paper_list}")
        return prompt

    def gen_prompt(self, popular_paper_list):
        """
        Generate the final prompt for the LLM.

        Args:
            popular_paper_list: List of recent popular papers.

        Returns:
            str: Final prompt for the LLM.
        """
        prompt = self._gen_search_term_prompt_by_paper(popular_paper_list)
        logger.info("Prompt generated successfully.")
        return prompt

    def run(self, popular_paper_list):
        """
        Execute the agent to analyze popular research trends.

        Args:
            popular_paper_list: List of recent popular papers.

        Returns:
            PopularResult: Result containing the analysis of popular research directions.
        """
        logger.info(f"Starting {self.task_name} with popular paper list: {popular_paper_list}")
        prompt = self.gen_prompt(popular_paper_list)
        popular_result = self._call_llm(
            self.system_msg,
            prompt,
            fail_times=3
        )
        logger.info(f"LLM call completed. Result: {popular_result}")
        return popular_result

    def _parse_data(self, llm_response, sys=None, prompt=None, llm_result=None):
        """
        Parse the response from the LLM into a PopularResult object.

        Args:
            llm_response: Response from the LLM.
            sys: System message (optional).
            prompt: Prompt used for the LLM (optional).
            llm_result: Raw result from the LLM (optional).

        Returns:
            PopularResult: Parsed result containing popular research directions and rationale.
        """
        logger.debug(f"Parsing LLM response: {llm_response}")
        return PopularResult(popular_research_directions=llm_response, rationale="",
                             sys=sys, prompt=prompt, llm_result=llm_result,
                             success=True)