import json

class MethodDecomResults:
    def __init__(
            self, method_decom_list, sys=None, prompt=None,
            llm_response=None, success=False):
        """
        method_decom_list: list of MethodDecom objects
        success: bool
        """
        self.method_decom_list = method_decom_list
        self.sys = sys
        self.prompt = prompt
        self.llm_response = llm_response
        self.success = success

    def to_dict(self):
        return {
            "method_decom_list":self.method_decom_list,
            "sys": self.sys,
            "prompt": self.prompt,
            "llm_response": self.llm_response,
            "success": self.success,
        }

    def __str__(self):
        return json.dumps(self.__dict__, indent=4, ensure_ascii=False)

    def get_method_decom_list(self):
        return self.method_decom_list

    def load_from_json(self, json_str):
        data = json.loads(json_str)
        self.method_decom_list = data['method_decom_list']
        self.success = data['success']
        return self

    @staticmethod
    def get_single_method_decom(method_decom):
        s = ''
        s += f"Module: {method_decom['Module']}\n"
        s += f"Purpose: {method_decom['Purpose']}\n"
        s += f"Components: {method_decom['Components']}\n"
        return s

    def get_architecture_of_the_method(self) -> str:
        s = ""
        for i, method_decom in enumerate(self.method_decom_list):
            s += f"Module {i+1}: {method_decom['Module']}\n"
            # s += f"Purpose: {method_decom['Purpose']}\n"
            # s += f"Components: {method_decom['Components']}\n"
        return s


    @staticmethod
    def gen_submethod_summary(sub_module:dict) -> str:
        """
        {
         "Module": f"Module {module_number}",
         "Purpose": purpose.strip(),
         "Components": components,
         "SearchKeyWords": keywords,
         "Sorted_document_with_score: 检索的top10的排序后的结果
         "SubMethodResult": ""
        }
         """
        s = ''
        s += f"SubModule: {sub_module['Module']}\n"
        s += "Detailed Submodule Design:" + sub_module['SubMethodResult'].split("Detailed Submodule Design:")[-1]
        return s


class SubMethodResult:
    def __init__(self, llm_result, success=False):
        self.llm_result = llm_result
        self.success = success

    def __str__(self):
        return json.dumps(self.__dict__, indent=4, ensure_ascii=False)

class FinalMethodResult:
    def __init__(self, llm_result, success=False):
        self.llm_result = llm_result
        self.success = success

    def __str__(self):
        return json.dumps(self.__dict__, indent=4, ensure_ascii=False)


