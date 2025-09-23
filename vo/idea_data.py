class IdeaResult:
    def __init__(self, thinking, idea, keywords, rationale=None, source=None, llm_result=None, prompt=None, root_source=None):
        self.thinking = thinking
        self.idea = idea
        self.keywords = keywords
        self.llm_result = llm_result
        self.prompt = prompt
        self.rationale = rationale
        self.source = source
        self.root_source = root_source

    def __str__(self):
        s = ""
        if self.thinking:
            s += f"Thinking: {self.thinking}"
        if self.idea:
            s += f"\nIdea: {self.idea}"
        if self.keywords:
            s += f"\nKeywords: {self.keywords}"
        return s

    @staticmethod
    def init_from_dict(data):
        return IdeaResult(
            thinking=data.get("thinking", None),
            idea=data.get("idea", None),
            keywords=data.get("keywords", None),
            rationale=data.get("rationale", None),
            source=data.get("source", None),
            root_source=data.get("root_source", None),
            llm_result=data.get("llm_result", None),
            prompt=data.get("prompt", None)
        )

    def to_dict(self):
        rt = dict()
        if self.thinking:
            rt["thinking"] = self.thinking
        if self.idea:
            rt["idea"] = self.idea
        if self.keywords:
            rt["keywords"] = self.keywords
        if self.rationale:
            rt["rationale"] = self.rationale
        if self.prompt:
            rt["prompt"] = self.prompt
        if self.llm_result:
            rt["llm_result"] = self.llm_result
        if self.source:
            rt["source"] = self.source
        if self.root_source is not None:
            rt['root_source'] = self.root_source
        return rt

def load_IdeaResult_from_dict(d):
    """load IdeaResult from dict"""
    thinking = d.get("thinking", "")
    idea = d.get("idea", "")
    keywords = d.get("keywords", "")
    rationale = d.get("rationale", "")
    source = d.get("source", "")
    prompt = d.get("prompt", "")
    llm_result = d.get("llm_result", "")
    return IdeaResult(thinking=thinking, idea=idea, keywords=keywords,
                         rationale=rationale, source=source,
                         llm_result=llm_result, prompt=prompt)

class SeedIdeaResult:
    def __init__(self,
                 seed_list,
                 paper_qa_info=None,
                 prompt=None,
                 llm_result=None,
                 success=False
                 ):
        self.paper_qa_info = paper_qa_info
        self.seed_list = seed_list
        self.prompt = prompt
        self.llm_result = llm_result
        self.success = success

    def __str__(self):
        s = ''
        if self.paper_qa_info:
            s += f"""# step1: QA\n{self.paper_qa_info}\n"""
        if self.seed_list:
            s += "# step2: new research directions\n"
            i = 0
            for seed in self.seed_list:
                s += "## Idea " + str(i) + "\n"
                s += str(seed) + "\n"
        return s

    def to_dict(self):
        return {
            "prompt": self.prompt,
            "llm_result": self.llm_result,
            "paper_qa_info": self.paper_qa_info,
            "seed_list": [seed.to_dict() for seed in self.seed_list]
        }

    def get_seed_list(self):
        return self.seed_list

def load_SeedIdeaResult_from_dict(d):
    """load SeedIdeaResult from dict"""
    paper_qa_info = d.get("paper_qa_info", None)
    prompt = d.get("prompt", None)
    llm_result = d.get("llm_result", None)
    paper_qa_info = d.get("paper_qa_info", None)
    seed_list = [load_IdeaResult_from_dict(seed) for seed in d.get("seed_list", [])]
    return SeedIdeaResult(seed_list, paper_qa_info, prompt, llm_result, True)

class SeedIdea:
    def __init__(self, paper_id, title, abstract, keywords, rationale, source):
        self.paper_id = paper_id
        self.title = title
        self.abstract = abstract


class SearchPlan:
    def __init__(self, plan_list, sys, prompt, llm_result, success=False):
        self.sys = sys
        self.prompt = prompt
        self.llm_result = llm_result
        self.plan_list = plan_list
        self.success = success

    def __str__(self):
        return f"""SearchPlan: {','.join(self.plan_list)}"""

    def get_search_plan(self):
        return self.plan_list

    def get_prompt(self):
        return self.prompt

    def get_llm_response(self):
        return self.llm_result

    def to_dict(self):
        return {
            "sys": self.sys,
            "prompt": self.prompt,
            "llm_result": self.llm_result,
            "plan_list": self.plan_list,
            "success": self.success
        }


class RagObs:
    def __init__(self, obs, success=False):
        self.obs = obs
        self.success = success

    def __str__(self):
        return f"""Obs: {','.join(self.obs)}"""

    def get_obs(self):
        return self.obs