import json
import logging
import os
from openai import OpenAI

# Initialize logger
logger = logging.getLogger()

class DeepseekLLM:
    """
    A class to interact with the Deepseek LLM (Language Learning Model) API.
    """

    def __init__(self, args):
        """
        Initializes the DeepseekLLM instance.

        Args:
            args: Arguments required for initialization.
        """
        self.args = args
        # Ensure the DEEPSEEK_API_KEY is set in the environment
        client = OpenAI(api_key=args.DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
        self.client = client
        logger.info("Initialized DeepseekLLM successfully!")

    def run(self, system_msg, question):
        """
        Sends a request to the Deepseek LLM API and returns the response.

        Args:
            system_msg (str): The system message to set the context for the LLM.
            question (str): The user's question or prompt.

        Returns:
            str: The response from the LLM.
        """
        # Create a chat completion request to the Deepseek LLM
        completion = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": question}
            ],
            temperature=0.3,
        )
        # Extract the response content
        res = completion.choices[0].message.content

        # Log the interaction details
        interaction_details = {
            "system_msg": system_msg,
            "prompt": question,
            "llm_response": res,
        }
        logger.info(f"DeepseekLLM Interaction | {json.dumps(interaction_details, ensure_ascii=False)}")

        return res