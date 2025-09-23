import logging
from agent.base_agent import BaseAgent
from vo.idea_data import IdeaResult
from utils_tool import extract_json_between_markers
import time

logger = logging.getLogger()

class IdeaIterationAgent(BaseAgent):
    """Generate new ideas based on search topic and plan then search result"""

    def __init__(self, args):
        BaseAgent.__init__(self, args)
        self.args = args
        self.system_msg = """"""
        self.task_name = "IdeaIterationAgent"
        self.idea_json_format = """In <JSON>, provide the new idea list in JSON format, every idea with the following fields:
            - "thinking": The thought process behind proposing the new idea, should explained the topic of target paper and old idea first, then explained what new insight you have get from the new knowledge to imporve current topic and the benefits.
            - "idea": detail valuable and new ideas worth exploring, make sure the idea follow the target paper's topic for further improvement and extend from old idea and The title should be about 10 words and should reflect the topic of the idea.
            - "keywords": keywords that can help search in google scholar then you can find related paper， keywords should separate with commas
            - "rationale": The reason for generating this idea. You can explain what challenge it solves, why this idea is innovative, and why it has the potential to develop into a proposal for a top conference..
            """
        # self.expand_idea_example = """
        # old_idea: "Utilize synthetic data generation techniques to create high-quality and diverse datasets for fine-tuning the LLM, improving the accuracy and relevance of the generated research ideas."
        # New Idea Example:
        # ```json\n[
        #     {
        #         "thinking": "the topic is using llm generate research ideas, Combine synthetic data with real-world data to create a hybrid dataset for fine-tuning LLM generate research ideas. This approach leverages the benefits of both data types to improve model performance.",
        #         "idea": "A hybrid synthetic data generation method that balances diversity and reliability for LLM generate research ideas.",
        #         "keywords": "hybrid data generation, synthetic and real-world data, use LLM generate research ideas, AI model fine-tuning, data integration",
        #         "rationale": "This idea addresses the challenge of relying solely on synthetic data by combining it with real-world data. The hybrid approach ensures a balance between data diversity and reliability, potentially leading to better model performance and more accurate research idea generation.",
        #     }
        #     ]\n```"""
       # Load demo examples from a file
        with open("prompts/idea_iteration_example.txt", "r") as f:
            self.expand_idea_example = f.read().strip()
        pass

    def _gen_prompt(self, paper, old_idea, new_knowledge, use_self_reflextion=True):
        # Generate a prompt for the LLM based on the provided paper, old idea, and new knowledge
        prompt = f"""
            # Role
            You are an expert researcher in AI. You can learn from new knowledge and provide some impactful and creative new ideas to improve old idea based on target paper.
            # Skills: Propose some innovative and valuable new research idea follow the following steps:
            1. Understand the research topic of target paper and old idea well, Analyze the possible optimization directions of the old idea.
            2. Understand the new knowledge well, Analyze the innovations, ideas and methods they have used, thinking what can use for propose new idea
            3. generate 3 most innovative and important idea in final for further improve old idea.
            """
        if use_self_reflextion:
            prompt += "Please generate 10 potential research ideas first, analyze the pros and cons of each idea, and finally select and generate 3 most innovative and important ideas to further improve the old ideas"
            prompt += f"In the thinking step, you can first think of about 10 new ideas and analyze the advantages and " \
                      f"disadvantages of each of them. Your final idea can absorb their advantages and discard their " \
                      f"disadvantages."
        prompt += f"""
            # Input Data Description
            It is important to understand target paper, old idea and new knowledge:
            - The target paper is the primary research study you aim to enhance or build upon through future research, serving as the central source and focus for identifying and developing the specific research idea.
            - The old idea is an existing idea. Your new idea should be different from old.
            - The new knowledge are some literature and paper you read when you study old idea. Your new idea should get inspiration from these new knowledge.
            # Format
            {self.idea_json_format}
            # Example
            Note: The following is an example of generating a new idea. You can refer to its reasoning process and output format, but do not use the insight in the example.            {self.expand_idea_example}
            # Requirements
            1. new idea should be related to target paper, follow the target paper's topic for further improvement
            2. The new idea is based on the old idea for further optimization based on the new knowledge.
            3. You should aim for new research ideas that can potentially win best paper awards at top conferences like ACL and NeurIPS and ICLR and CVPR.\
            4. The new knowledge are only for inspiration and you should not cite them and just make some incremental modifications. Instead, you should make sure your ideas are novel and distinct from the prior literature.
            5. Please output your thought process
            6. Only the final three ideas should be JSON format
            7. Please note that you should not use the idea in the example. Please come up with novel and feasible ideas based on inputs bellow (target paper and context, etc.).
            8. please thinking step by step
            # Input
            target paper title:{paper.title}
            target paper abstract:{paper.abstract}
            old_idea:{old_idea}
            new knowledge:{new_knowledge}
            # Output
            ### Thinking
            <output your thinking process here, Explain the target paper topic, explain how you improved an old idea, explain what new knowledge you used for new idea generation and why it make sense and why it should have change to win the best paper awards at top conferences>
            ### 10 possible new ideas(weakness and strenth analysis)
            <MarkDownFormatFor10PossibleResult>
            ### Final Three Most Innovative and Important Ideas Base On Old Idea
            <JSON>
            """
        return prompt

    def parse_seed_idea_data(self, response, source="expand_idea"):
                # Parse the response to extract idea data in JSON format
        idea_result_list= []
        idea_dict_list = extract_json_between_markers(response)
        for idea_dict in idea_dict_list:
            if isinstance(idea_dict, dict):
                idea = IdeaResult.init_from_dict(idea_dict)
                idea.source = source
                idea_result_list.append(idea)
        return idea_result_list

    def run(self, paper, cur_idea, related_paper_about_cur_idea):
        """Generate new ideas based on the current paper, current idea, and related papers
        Args:
            paper: The target paper
            cur_idea: The current idea
            related_paper_about_cur_idea: Related papers about the current idea
        """
        prompt = self._gen_prompt(paper, cur_idea, related_paper_about_cur_idea)
        for try_cnt in range(3):
            try:
                llm_result = self._call_llm(
                    self.system_msg,
                    prompt,
                    fail_times=3,
                    parse_data_flag=False
                )
                idea_list = self.parse_seed_idea_data(llm_result)
                return prompt, llm_result, idea_list
            except:
                time.sleep(try_cnt)
