import json
import logging
import os
from dotenv import load_dotenv
from openai import OpenAI
from openai import AzureOpenAI
from src.ai_researcher_tool import calc_price

# Initialize logger
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

class OpenaiLLM:
    """
    A class to interact with OpenAI or Azure OpenAI services.
    Defaults to Azure OpenAI if no specific mode is provided.
    """

    def __init__(self, args):
        """
        Initialize the OpenaiLLM class with the provided arguments.

        Args:
            args (argparse.Namespace): Arguments containing necessary configurations.
        """
        self.args = args

        # Initialize the client based on the mode (Azure OpenAI or Native OpenAI)
        if getattr(args, 'use_native_openai', False):
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable is not set. Please set it in your .env file or environment.")
            self.client = OpenAI(api_key=api_key)
            self.MODEL_NAME = self.args.NATIVE_OPENAI_MODEL_NAME
            logger.info("Initialized OpenAI LLM in Native OpenAI mode.")
        else:
            if not args.AZURE_OPENAI_ENDPOINT:
                raise ValueError("AZURE_OPENAI_ENDPOINT is not set in config. Please configure it in your config file.")
            if not args.AZURE_OPENAI_KEY:
                raise ValueError("AZURE_OPENAI_KEY is not set in config. Please configure it in your config file.")
            if not args.AZURE_OPENAI_API_VERSION:
                raise ValueError("AZURE_OPENAI_API_VERSION is not set in config. Please configure it in your config file.")
            self.client = AzureOpenAI(
                azure_endpoint=args.AZURE_OPENAI_ENDPOINT,
                api_key=args.AZURE_OPENAI_KEY,
                api_version=args.AZURE_OPENAI_API_VERSION
            )
            self.MODEL_NAME = self.args.AZURE_OPENAI_MODEL_NAME
            logger.info("Initialized OpenAI LLM in Azure OpenAI mode.")

        logger.info("OpenaiLLM initialization completed.")

    def run(self, system_msg, prompt):
        """
        Run the LLM with the provided system message and user question.

        Args:
            system_msg (str): The system message to set the context.
            question (str): The user's question or prompt.

        Returns:
            str: The response from the LLM.
        """
        res = None
        try:
            assert len(prompt) > 0, "Question cannot be empty."

            # Prepare the messages for the LLM
            if system_msg:
                messages = [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt}
                ]
            else:
                messages = [
                    {"role": "user", "content": prompt}
                ]

            # Call the LLM API
            completion = self.client.chat.completions.create(
                model=self.MODEL_NAME,  # Updated to "gpt-4o"
                messages=messages,
                temperature=getattr(self.args, 'temperature', 0.3),
            )

            # Calculate the cost of the API call
            cost = calc_price(model=self.MODEL_NAME, usage=completion.usage)  # Updated to "gpt-4o"

            # Extract the response from the completion
            res = completion.choices[0].message.content

            # Log the interaction details
            interaction_details = {
                "system_msg": system_msg,
                "prompt": prompt,
                "llm_response": res,
                'cost': cost,
                "success": True,
                "model": "openai_"+self.MODEL_NAME
            }
            logger.info(f"OpenaiLLM Interaction | {json.dumps(interaction_details, ensure_ascii=False)}")
        except Exception as e:
            error_msg = str(e)
            logger.error(f"OpenaiLLM | API call failed: {error_msg}")
            interaction_details = {
                "system_msg": system_msg,
                "prompt": prompt,
                "llm_response": res,
                "success": False,
                'cost': 0,
                "msg": error_msg,
                "model": "openai_"+self.MODEL_NAME
            }
            logger.info(f"OpenaiLLM Interaction | {json.dumps(interaction_details, ensure_ascii=False)}")
        return res


if __name__ == '__main__':

    from args_tool import get_args
    args = get_args()
    llm = OpenaiLLM(args)

    print(llm.run(
        system_msg="",
        prompt="hi, 1+1=?"
    ))
