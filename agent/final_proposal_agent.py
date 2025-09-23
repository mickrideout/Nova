import logging
import time
from agent.base_agent import BaseAgent
import vo.paper_data
from src.ai_researcher_tool import format_plan_json
from utils_tool import extract_json_between_markers

# Initialize logger
logger = logging.getLogger(__name__)


class FinalProposalGenerator(BaseAgent):
    """
    Agent responsible for generating a final research proposal based on the given idea, method decomposition, feedback, and new knowledge.
    """

    def __init__(self, args):
        """
        Initialize the FinalProposalGenerator with the given arguments.

        Args:
            args: Configuration arguments for the agent.
        """
        super().__init__(args)  # Initialize the base class
        self.args = args
        self.system_msg = ""  # System message for LLM interaction
        self.task_name = "FinalProposalGenerator"  # Name of the task for logging
        # Load demo examples from a file
        # with open("prompts/experiment_plan_examples_prompting_cleaned.txt", "r") as f:
            # self.demo_examples = f.read().strip()
        with open("prompts/final_proposal_example.txt", "r") as f:
            self.demo_examples = f.read().strip()
        logger.info(f"{self.task_name} | Initialized with demo examples loaded")

    def gen_prompt(
            self,
            paper: vo.paper_data.Paper,
            initial_proposal: dict,
            method_decom_info: str,
            feedback: str,
            new_knowledge: str
    ) -> str:
        """
        Generate the prompt for creating the final research proposal.

        Args:
            paper (vo.paper_data.Paper): The target paper to enhance or build upon.
            initial_proposal (dict): The initial research idea containing problem, existing methods, proposed method, and experiment plan.
            method_decom_info (str): Optimized method module design, decomposed into submodules.
            feedback (str): Feedback on the initial idea or method design.
            new_knowledge (str): New knowledge introduced by the submodules.

        Returns:
            str: The generated prompt for the LLM.
        """
        prompt = "You are an expert researcher in AI and your job is to expand a brief project idea into a full project proposal with detailed methodology and experiment plans so that your students can follow the steps and execute the full project. \n\n"
        # prompt += "The idea is:\n" + format_plan_json(initial_proposal) + "\n"
        prompt += "The target paper is the primary research study you aim to enhance or build upon through future research, serving as the central source and focus for identifying and developing the specific research idea."
        # prompt += f"target paper title:{paper.title}, target paper abstract:{paper.abstract}"

        prompt += "Now you should come up with the full proposal covering:\n"
        prompt += "1. Title: A concise statement of the main research question to be used as the paper title.\n"
        prompt += "2. Problem Statement: Clearly define the problem your research intends to address. Explain clearly why this problem is interesting and important.\n"
        prompt += "3. Motivation: Explain why existing methods (both classic ones and recent ones) are not good enough to solve the problem, and explain the inspiration behind the new proposed method. You should also motivate why the proposed method would work better than existing baselines on the problem.\n"
        prompt += "4. Proposed Method: Detail Explain how the proposed method works, describe all the steps. Make sure every step is clearly described and feasible to implement.\n"
        prompt += "5. Step-by-Step Experiment Plan: Break down every single step of the experiments, make sure every step is executable. Cover all essential details such as the datasets, models, and metrics to be used. If the project involves prompting, give example prompts for each step.\n"
        prompt += "The experiment plan should not include any background introduction (you can skip the literature review, paper writing tips, and ethical discussion). Just give instructions on the experiments.\n"
        prompt += "Be consistent in your methodology and experiment design, for example, if you will use black-box LLM APIs such as GPT and Claude for your experiments, then you shouldn't propose any experiments that require white-box model weights or data access and you should edit them accordingly to follow the black-box assumptions.\n"
        if self.demo_examples:
            prompt += "# OutputExample"
            prompt += "Note that we only provide examples related to prompt work. Please refer to this format in other fields, You can follow these examples to get a sense of how the idea should be formatted (but don't bore the ideas), Below are a few examples of how the full experiment plans should look like:\n"
            prompt += self.demo_examples + "\n\n"
        prompt += """# Requirements
            1. Consider novelty, significance, correctness, and reproducibility to ensure the high quality of the final proposal
            2. You should aim for final proposal that can potentially win best paper awards at top conferences like ACL and NeurIPS and ICLR and CVPR.\
            3. first give an overview of the proposed method, then give a detailed design
            4. Please output your thought process
            5. please thinking step by step
            """
        prompt += "Now please write down your final proposal in JSON format (keys should be the section names, just like the above examples). Make sure to be as detailed as possible so that a student can directly follow the plan to implement the project."
        prompt += "# Input\n\n"
        prompt += f"## Target Paper\n\n target paper title:{paper.title}, target paper abstract:{paper.abstract}"
        prompt += "## Initial proposal\n\n" + format_plan_json(initial_proposal) + "\n"
        if method_decom_info:
            prompt += "## MethodSubmodulesJsonList\n\n"
            prompt += "You will be given a method_decom_info(possible submethod module design). You need to analyze whether the design of these modules is reasonable. Please learn from the good places to complete the detailed design of the final method.."
            prompt += f"method_decom_info:{method_decom_info}"
        if feedback:
            prompt += "## Feedback\n\n"
            prompt += f"The feedback is: {feedback} \n"
        if new_knowledge:
            prompt += "## New Knowledge\n\n"
            prompt += f"The new knowledge is:{new_knowledge} \n"
        prompt += """
            # Output
            ## Thinking:
            <output your thinking process here, explain why it should have change to win the best paper awards at top conferences
            ## FinalProposal:
            <JSON>
        """
        logger.info(f"{self.task_name} | Prompt generated for paper: {paper.title}")
        return prompt

    def run(self,
            paper: vo.paper_data.Paper,
            initial_proposal: dict,
            method_decom_info: str,
            feedback: str = None,
            new_knowledge: str = None
            ) -> dict:
        """
        Execute the final proposal generation process.

        Args:
            paper (vo.paper_data.Paper): The target paper to enhance or build upon.
            initial_proposal (dict): The initial research idea.
            method_decom_info (str): Optimized method module design, decomposed into submodules.
            feedback (str, optional): Feedback on the initial idea or method design. Defaults to None.
            new_knowledge (str, optional): New knowledge introduced by the submodules. Defaults to None.

        Returns:
            dict: The final proposal, prompt, and LLM response.
        """
        for try_cnt in range(3):
            try:
                prompt = self.gen_prompt(paper, initial_proposal, method_decom_info, feedback, new_knowledge)
                logger.info(f"{self.task_name} | Starting final proposal generation for paper: {paper.title}")
                llm_response = self._call_llm(
                    self.system_msg,
                    prompt,
                    fail_times=3,
                    parse_data_flag=False
                )
                result = {
                    "final_proposal": extract_json_between_markers(llm_response),
                    "prompt": prompt,
                    "llm_response": llm_response
                }
                logger.info(f"{self.task_name} | Final proposal generated successfully for paper: {paper.title}")
                return result
            except Exception as e:
                time.sleep(try_cnt)
                logger.error(f"{self.task_name} | Failed {try_cnt + 1} times, error: {e}")
        return {"final_proposal": None, "prompt": None, "llm_response": None}

    def _parse_data(self, llm_response: str, sys=None, prompt=None, llm_result=None) -> str:
        """
        Parse the LLM response to extract the final proposal.

        Args:
            llm_response (str): The response from the LLM.
            sys (str, optional): System message used in the LLM call. Defaults to None.
            prompt (str, optional): Prompt used in the LLM call. Defaults to None.
            llm_result (str, optional): Raw result from the LLM. Defaults to None.

        Returns:
            str: The parsed final proposal.
        """