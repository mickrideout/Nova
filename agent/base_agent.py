import logging
from vo.base import BaseVO
from llm_adpter.openai_llm import OpenaiLLM
from llm_adpter.deepseek_llm import DeepseekLLM

# Initialize logger
logger = logging.getLogger()

class BaseAgent:
    """
    Base class for agents that interact with different LLM models.
    """

    def __init__(self, args):
        """
        Initialize the BaseAgent with the given arguments.

        Args:
            args: Configuration arguments for the agent.
        """
        self.args = args
        self.task_name = "BaseAgent"
        # Initialize the appropriate LLM based on the model specified in args
        if self.args.LLM_COMPANY == 'openai':
            self.llm = OpenaiLLM(args)
            logger.debug(f"Initialized OpenAI LLM for {self.task_name}")
        elif self.args.LLM_COMPANY == 'deepseek':
            self.llm = DeepseekLLM(args)
            logger.debug(f"Initialized Deepseek LLM for {self.task_name}")
        else:
            logger.error(f"Unsupported model: {args.model}")
            raise NotImplementedError(f"Model {args.model} is not supported")

    def _call_llm(self, sys, prompt, fail_times=3, parse_data_flag=True):
        """
        Call the LLM model and parse the response if required.

        Args:
            sys: System message or context for the LLM.
            prompt: The input prompt for the LLM.
            fail_times: Maximum number of retries on failure.
            parse_data_flag: Flag to indicate whether to parse the LLM response.

        Returns:
            BaseVO: Result object indicating success or failure.
        """
        i = 0
        while i < fail_times:
            i += 1
            try:
                # Attempt to call the LLM
                res = self.llm.run(sys, prompt)
                logger.info(f"{self.task_name} | LLM call successful on attempt {i}")

                # Parse the response if required
                if parse_data_flag:
                    result = self._parse_data(res)
                    if result.success:
                        logger.info(f"{self.task_name} | Data parsed successfully")
                        return result
                    else:
                        logger.warning(f"{self.task_name} | Data parsing failed")
                else:
                    return res
            except Exception as e:
                logger.error(f"{self.task_name} | LLM call failed on attempt {i}, error: {e}")

        # Return failure if all attempts are exhausted
        logger.error(f"{self.task_name} | All LLM call attempts failed")
        return BaseVO(success=False)

    def _parse_data(self, llm_response):
        """
        Parse the response from the LLM.

        Args:
            llm_response: The response from the LLM.

        Raises:
            NotImplementedError: This method should be implemented by subclasses.
        """
        logger.error(f"{self.task_name} | _parse_data method not implemented")
        raise NotImplementedError("_parse_data method must be implemented by subclass")