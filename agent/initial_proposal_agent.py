import logging
import vo.paper_data
from vo.paper_data import Paper
from agent.base_agent import BaseAgent
import re
import json
from src.ai_researcher_tool import shuffle_dict_and_convert_to_string
from utils_tool import extract_json_between_markers, load_json_from_file, save_json_data_to_file

# Initialize logger
logger = logging.getLogger()

class InitialProposalAgent(BaseAgent):
    """Agent to generate initial proposal based on given idea and other information."""

    def __init__(self, args):
        """Initialize the InitialProposalAgent with given arguments."""
        BaseAgent.__init__(self, args)
        self.args = args
        self.system_msg = """"""
        self.task_name = "InitialProposalAgent"
        self.method_proposal_examples = self.load_examples()

    def load_examples(self):
        """Load examples from JSON files and shuffle them."""
        examples = dict()
        try:
            with open("prompts/idea_examples_prompting_method.json", "r") as f:
                method_idea_examples = json.load(f)
                examples.update(method_idea_examples)
            with open("prompts/idea_examples_finetuning_method.json", "r") as f:
                method_idea_examples = json.load(f)
                examples.update(method_idea_examples)
            method_idea_examples = shuffle_dict_and_convert_to_string(examples, int(len(examples) // 2))
            logger.info("Examples loaded and shuffled successfully.")
        except Exception as e:
            logger.error(f"Failed to load examples: {e}")
            raise
        return method_idea_examples

    def gen_prompt(self, paper: Paper, seed_idea: dict, search_plan_result_list:list, use_few_shot_example: bool = True, use_self_reflextion=True) -> str:
        """Generate a prompt for the LLM based on the given paper and idea."""
        # 1. Get the idea string
        idea = f"""idea:{seed_idea['idea']},
                            keywords:{str(seed_idea["keywords"])},
                            thinking:{seed_idea['thinking']}"""
        # 2. Get the retrieval knowledge by plan retrieval
        retrieval_papers = ""
        for search_plan_result in search_plan_result_list:
            retrieval_papers += "Search_info:" + search_plan_result["search_info"]
            retrieval_papers += "Search_result:" + search_plan_result["top1_document"]
        prompt = "You are an expert researcher in Large Language Models. Now I want you to help me brainstorm the detail research project proposal base on the idea: " + idea + ".\n\n"
        prompt += f"the giving idea are driving from paper:{paper.title}, abstract:{paper.abstract}, this just for your background knowledge:\n""\n"
        prompt += "Here are some relevant papers on this idea just for your background knowledge:\n" + retrieval_papers + "\n"
        prompt += "You should generate detail proposal base on the given knowledge. Try to be creative. "
        prompt += "The above papers are only for inspiration and you should not cite them and just make some incremental modifications. Instead, you should make sure your proposal are novel and distinct from the prior literature. "
        prompt += "You should aim for projects that can potentially win best paper awards at top conferences like ACL and NeurIPS.\n"
        prompt += "The poposal should be described as: " \
                  "(1) Problem: State the problem statement, which should be closely related to the idea description and something that large language models cannot solve well yet. " \
                  "(2) Existing Methods: Mention some existing benchmarks and baseline methods if there are any, you should summrise the Existing Methods from target paer and relevant papers, not idea. " \
                  "(3) Motivation: Explain the inspiration of the proposed method and why it would work well. " \
                  "(4) Proposed Method: Propose your new method and describe it in detail. The proposed method should be maximally different from all existing work and baselines, and be more advanced and effective than the baselines. You should be as creative as possible in proposing new methods, we love unhinged ideas that sound crazy. This should be the most detailed section of the proposal. " \
                  "(5) Experiment Plan: Specify the experiment steps, baselines, and evaluation metrics.\n"
        if use_few_shot_example:
            prompt += "You can follow these examples to get a sense of how the proposal should be formatted (but don't borrow the proposal themselves):\n" + self.method_proposal_examples + "\n"
        prompt += "You should make sure to come up with your own novel proposal for the specified idea: " + idea + ". You should try to tackle important problems that are well recognized in the field and considered challenging for current models. For example, think of novel solutions for problems with existing benchmarks and baselines. In rare cases, you can propose to tackle a new problem, but you will have to justify why it is important and how to set up proper evaluation.\n"
        # if use_self_reflextion:
        #     prompt += "In the thinking step, you can first think of about 5 proposal and analyze the advantages and disadvantages of each of them. Your final proposal can absorb their advantages and discard their disadvantages."
        prompt += "attention proposal show base on the idea: " + seed_idea['idea'] + ".\n\n"
        prompt += f"and topic should follow the paper:{paper.title}, abstract:{paper.abstract}\n""\n"
        prompt += "Please write down thinking and your final proposal. Output the final proposal in json format as a dictionary, where you should generate a short proposal name (e.g., \"Non-Linear Story Understanding\", or \"Multi-Agent Negotiation\") as the key and the actual proposal description as the value (following the above format). "
        prompt += """# Output:
                    Thinking: our topic is ..., our idea is ....
                    proposal:
                """
        logger.info("Prompt generated successfully.")
        return prompt

    def run(self, paper: Paper, seed_idea: dict, search_plan_result_list:list):
        """
        Run the InitialProposalAgent to generate an initial proposal.

        Args:
            paper (vo.paper_data.Paper): Target paper
            seed_idea (dict): Idea information
            search_plan_result_list (list): Search plan result list

        Returns:
            dict: Result containing the proposal, prompt, and LLM response
        """
        for try_cnt in range(3):
            try:
                prompt = self.gen_prompt(paper=paper, seed_idea=seed_idea, search_plan_result_list=search_plan_result_list)
                llm_response = self._call_llm(
                    self.system_msg,
                    prompt,
                    fail_times=3,
                    parse_data_flag=False
                )
                initial_proposal = extract_json_between_markers(llm_response)
                if initial_proposal is None:
                    logger.info(f"InitialProposalAgent | initial_proposal is None. try_cnt:{try_cnt}, initial_proposal:{initial_proposal}")
                    continue

                if "Title" in initial_proposal:
                    key = initial_proposal['Title']
                    value = initial_proposal
                    initial_proposal = {key: value}
                assert len(initial_proposal) == 1, f"len(initial_proposal):{len(initial_proposal)}"
                idea_k = list(initial_proposal.keys())[0]
                idea_v = initial_proposal[idea_k]
                assert "Problem" in idea_v, f"Problem:paper:{paper.title},v:{idea_v}"
                assert "Existing Methods" in idea_v, f"Existing Methods: paper:{paper.title},v:{idea_v}"
                assert "Motivation" in idea_v, f"Motivation: paper:{paper.title},v:{idea_v}"
                assert "Proposed Method" in idea_v, f"Proposed Method: paper:{paper.title},v:{idea_v}"
                assert "Experiment Plan" in idea_v, f"Experiment Plan: paper:{paper.title},v:{idea_v}"

                result = {
                    "proposal": initial_proposal,
                    "prompt": prompt,
                    "llm_response": llm_response
                }
                logger.info("Proposal generated successfully.")
                return result
            except Exception as e:
                logger.error(f"InitialProposalAgent | gen initial_proposal error, try_cnt:{try_cnt}, e:{e}")
                import time
                time.sleep(try_cnt)

    def _parse_data(self, llm_response: str, sys=None, prompt=None, llm_result=None):
        """Parse the data from the LLM response."""
        pass