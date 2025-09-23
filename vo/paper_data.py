import json


class Paper:

    def __init__(self, title, abstract, related_paper_titles, related_paper_abstract, entities, categories = [], id = None):
        self.title = title
        self.abstract = abstract
        self.related_paper_titles = related_paper_titles
        self.related_paper_abstract = related_paper_abstract
        self.entities = entities
        self.categories = categories
        self.id = id


    def __str__(self):
        return json.dumps(
            {
                "title": self.title,
                "abstract": self.abstract,
                "related_paper_titles": self.related_paper_titles,
                "related_paper_abstract": self.related_paper_abstract,
                "entities": self.entities,
                "categories": self.categories,
                "id": self.id
            },
            ensure_ascii=False
        )

    def _to_dict(self):
        return {
                "title": self.title,
                "abstract": self.abstract,
                "related_paper_titles": self.related_paper_titles,
                "related_paper_abstract": self.related_paper_abstract,
                "entities": self.entities,
                "categories": self.categories,
                "id": self.id
            }

    def get_dict_result(self):
        return self._to_dict()

def load_paper_from_file(paper_file):
    """加载paper数据"""
    with open(paper_file, 'r') as fr:
        data = json.load(fr)
        return Paper(
            title=data['title'],
            abstract=data['abstract'],
            related_paper_titles=data['related_paper_titles'],
            related_paper_abstract=data['related_paper_abstract'],
            entities=data.get('entities', None),
            categories=data.get('categories', None),
            id=data.get('id', None)
        )

def load_paper_from_dict(paper_dict):
    """加载paper数据"""
    return Paper(
        title=paper_dict['title'],
        abstract=paper_dict['abstract'],
        related_paper_titles=paper_dict['related_paper_titles'],
        related_paper_abstract=paper_dict['related_paper_abstract'],
        entities=paper_dict.get('entities', None),
        categories=paper_dict.get('categories', None),
        id=paper_dict.get('id', None)
    )



class ProblemMethodOfPaper:

    def __init__(self, title, abstract, problem, method, llm_prompt, llm_response, success):
        self.title = title
        self.abstract = abstract
        self.problem = problem
        self.method = method
        self.llm_prompt = llm_prompt
        self.llm_response = llm_response
        self.success = success

    def to_dict(self):
        return {
            "title": self.title,
            "abstract": self.abstract,
            "problem": self.problem,
            "method": self.method,
            "llm_prompt": self.llm_prompt,
            "llm_response": self.llm_response,
            "success": self.success
        }

    def to_json(self):
        return json.dumps(self.to_dict())

    @staticmethod
    def from_dict(data):
        return ProblemMethodOfPaper(
            title=data['title'],
            abstract=data['abstract'],
            problem=data['problem'],
            method=data['method'],
            llm_prompt=data['llm_prompt'],
            llm_response=data['llm_response'],
            success=data['success']
        )

    @staticmethod
    def from_json(json_str):
        data = json.loads(json_str)
        return ProblemMethodOfPaper.from_dict(data)

