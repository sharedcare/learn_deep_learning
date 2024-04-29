import json
import logging
from typing import List, Optional, Tuple

import requests

from autogen.agentchat.agent import Agent
from autogen.agentchat.contrib.multimodal_conversable_agent import MultimodalConversableAgent
from autogen.code_utils import content_str

try:
    from termcolor import colored
except:
    def colered(x, *args, **kwargs):
        return x


logger = logging.getLogger(__name__)

# we will override the following variables later.
SEP = "###"

DEFAULT_LLAVA_SYS_MSG = "You are an AI agent and you can view images."

def cog_vlm_call(messages: list, config: dict, max_new_tokens: int = 1000, temperature: float = 0.8, top_p: float = 0.8, use_stream: bool = False):
    """
    This function sends a request to the chat API to generate a response based on the given messages.
    Refer to: https://github.com/THUDM/CogVLM/blob/main/openai_demo/openai_api_request.py

    Args:
        messages (list): A list of message dictionaries representing the conversation history.
        config (dict): The config of the llm model to use for generating the response.
        max_new_tokens (int): The maximum length of the new tokens from generated response.
        temperature (float): Controls randomness in response generation. Higher values lead to more random responses.
        top_p (float): Controls diversity of response by filtering less likely options.
        use_stream (bool): Determines whether to use a streaming response or a single response.

    The function constructs a JSON payload with the specified parameters and sends a POST request to the API.
    It then handles the response, either as a stream (for ongoing responses) or a single message.
    """
    headers = {"User-Agent": "CogAgent Client"}
    data = {
        "model": config["model"],
        "messages": messages,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        # "stop": SEP,
    }
    
    response = requests.post(
            config["base_url"].rstrip("/") + "/v1/chat/completions", headers=headers, json=data, stream=use_stream
        )
    
    if response.status_code == 200:
        if use_stream:
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')[6:]
                    try:
                        response_json = json.loads(decoded_line)
                        output = response_json.get("choices", [{}])[0].get("delta", {}).get("content", "")
                    except:
                        print("Special Token:", decoded_line)
        else:
            decoded_line = response.json()
            output = decoded_line.get("choices", [{}])[0].get("message", "").get("content", "")
    else:
        print("Error:", response.status_code)
        return None

    return output
    


class CogVLMAgent(MultimodalConversableAgent):
    def __init__(
        self,
        name: str,
        system_message: Optional[Tuple[str, List]] = DEFAULT_LLAVA_SYS_MSG,
        *args,
        **kwargs,
    ):
        """
        Args:
            name (str): agent name.
            system_message (str): system message for the ChatCompletion inference.
                Please override this attribute if you want to reprogram the agent.
            **kwargs (dict): Please refer to other kwargs in
                [ConversableAgent](../conversable_agent#__init__).
        """
        super().__init__(
            name,
            system_message=system_message,
            *args,
            **kwargs,
        )

        assert self.llm_config is not None, "llm_config must be provided."
        self.register_reply([Agent, None], reply_func=CogVLMAgent._image_reply, position=2)

    def _image_reply(self, messages=None, sender=None, config=None):
        # Note: we did not use "llm_config" yet.

        if all((messages is None, sender is None)):
            error_msg = f"Either {messages=} or {sender=} must be provided."
            logger.error(error_msg)
            raise AssertionError(error_msg)

        if messages is None:
            messages = self._oai_messages[sender]

        # The formats for CogVLM and GPT are different. So, we manually handle them here.
        out = ""
        retry = 10
        config = self.llm_config["config_list"][0]
        while len(out) == 0 and retry > 0:
            # image names will be inferred automatically from cog_vlm_call
            out = cog_vlm_call(
                messages=messages,
                config=config,
                max_new_tokens=config.get("max_new_tokens", 2000),
                temperature=config.get("temperature", 0.8),
                use_stream=False,
            )
            retry -= 1

        assert out != "", "Empty response from CogVLM."

        return True, out