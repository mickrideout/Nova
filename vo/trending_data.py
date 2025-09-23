class PopularResult:
    def __init__(self, popular_research_directions, rationale, sys, prompt, llm_result, success=True):
        self.popular_research_directions = popular_research_directions
        self.rationale = rationale
        self.success = success
        self.sys = sys
        self.prompt = prompt
        self.llm_result = llm_result

    def __str__(self):
        return f"""Popular_Research_Directions: {self.popular_research_directions}\nRationale:{self.rationale}"""

    def get_result(self):
        return self.popular_research_directions

    def get_rationale(self):
        return self.rationale


    def to_dict(self):
        return {
            "popular_research_directions": self.popular_research_directions,
            "rationale": self.rationale,
            "sys": self.sys,
            "prompt": self.prompt,
            "llm_result": self.llm_result,
            "success": self.success
        }
