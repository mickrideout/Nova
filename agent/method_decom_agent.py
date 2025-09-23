import logging
import vo.paper_data
from agent.base_agent import BaseAgent
from vo.method_decom_data import MethodDecomResults
from src.ai_researcher_tool import format_plan_json
from utils_tool import extract_json_between_markers

# Initialize logger
logger = logging.getLogger()

class MethodDecomAgent(BaseAgent):
    """
    Agent responsible for decomposing a research method into multiple submodules.
    """

    def __init__(self, args):
        """
        Initialize the MethodDecomAgent with the given arguments.

        Args:
            args: Configuration arguments for the agent.
        """
        super().__init__(args)  # Initialize the base class
        self.args = args
        self.system_msg = ""  # System message for LLM interaction
        self.task_name = "MethodDecomAgent"  # Name of the task for logging
        self.example = """
        ```json
        {
            "Module 1": {
            "title": "Real-Time Feedback Collection and Processing",
            "purpose": "To capture and process user feedback in real-time during or after the image generation process.",
            "implementation": {
              "User Interaction Interface": "Allows users to provide immediate feedback on the generated images (e.g., rating, comments, or specific adjustments).",
              "Feedback Interpretation Engine": "Analyzes the feedback to understand user preferences, translating them into actionable data for the image generation model.",
              "Continuous Learning Mechanism": "Uses the feedback to dynamically update the model or adjust subsequent outputs."
            },
            "search_keywords": [
              "real-time interaction systems",
              "adaptive systems with user feedback",
              "feedback-driven image generation",
              "continuous learning from user feedback"
            ]
        }
        ```
        """  # Example JSON format for submodules
        self.submodule_json_format = """
        In <JSON>, provide the new submodule list in JSON format, every submodule with the following fields:
            - "thinking": Explain your thought process in designing this submodule, why this design is reasonable, common sense, innovative, and feasible.
            - "title": The name of the module, describing its primary function or theme.
            - "purpose": The objective of the module, explaining the main goals for its design and implementation.
            - "implementation": Detailed description of the implementation of this submodule.
            - "search_keywords": keywords that can help search in google scholar then you can find related paper.
        """  # JSON format specification for submodules

    def _gen_prompt(self, paper: vo.paper_data.Paper, initial_proposal: dict) -> str:
        """
        Generate the prompt for decomposing the research method into submodules.

        Args:
            paper (vo.paper_data.Paper): The target paper.
            initial_proposal (dict): The initial_proposal containing problem, existing methods, proposed method, and experiment plan.

        Returns:
            str: The generated prompt for the LLM.
        """
        prompt = f"""
            # Role
            You are an expert researcher in AI, You can break down the research method into multiple submodules.
            # Task
            Give you a initial_proposal, You need to break down the method into separate modules, and you need to explain in detail the specific content, purpose, composition and keywords of each module (keywords are used to find relevant papers on google scholar).
            initial_proposal includes Problem, Existing Methods, Proposed Method, Experiment Plan, I will give you the initial_proposal:.
            {format_plan_json(initial_proposal)}"""
        if self.example:
            prompt += "here is an example, You can follow these examples to get a sense of how it should be formatted (but don't borrow the examples themselves):"
            prompt += self.example
        prompt += f"""
            # Requirements
            1. You only need to break down the method part of the idea into multiple sub-modules for detailed design. Your design cannot conflict with common sense and must be innovative, reasonable, and feasible.
            2. You should aim for the research ideas that can potentially win best paper awards at top conferences like ACL and NeurIPS and ICLR and CVPR.
            3. Please output your thought process
            4. please thinking step by step"""
        prompt += self.submodule_json_format
        prompt += f"""
            # Input
            initial_proposal:{format_plan_json(initial_proposal)}
            # Output
            Thinking:
            <output your thinking process here, explain why you choose these theory to discover new idea and why it should have change to win the best paper awards at top conferences
            MethodSubmodulesJsonList:
            <JSON>
            """
        logger.info(f"{self.task_name} | Prompt generated for paper: {paper.title}")
        return prompt

    def run(self, paper: vo.paper_data.Paper, initial_proposal: dict) -> MethodDecomResults:
        """
        Execute the method decomposition process.

        Args:
            paper (vo.paper_data.Paper): The target paper.
            initial_proposal (dict): The initial_proposal containing problem, existing methods, proposed method, and experiment plan.

        Returns:
            MethodDecomResults: The result of the method decomposition.
        """
        prompt = self._gen_prompt(paper, initial_proposal)
        logger.info(f"{self.task_name} | Starting method decomposition for paper: {paper.title}")
        decom_result = self._call_llm(
            self.system_msg,
            prompt,
            fail_times=3
        )
        logger.info(f"{self.task_name} | Method decomposition completed for paper: {paper.title}")
        return decom_result

    def _parse_data(self, llm_response, sys=None, prompt=None, llm_result=None) -> MethodDecomResults:
        """
        Parse the LLM response to extract the method decomposition results.

        Args:
            llm_response (str): The response from the LLM.
            sys (str, optional): System message used in the LLM call. Defaults to None.
            prompt (str, optional): Prompt used in the LLM call. Defaults to None.
            llm_result (str, optional): Raw result from the LLM. Defaults to None.

        Returns:
            MethodDecomResults: The parsed method decomposition results.
        """
        method_decom_list = extract_json_between_markers(llm_response)
        if len(method_decom_list) < 1:
            logger.warning(f"{self.task_name} | No valid JSON found in LLM response")
            return MethodDecomResults(method_decom_list=method_decom_list, success=False)
        logger.info(f"{self.task_name} | Successfully parsed method decomposition results")
        return MethodDecomResults(
            method_decom_list=method_decom_list,
            sys=sys,
            llm_response=llm_response,
            prompt=prompt,
            success=True
        )