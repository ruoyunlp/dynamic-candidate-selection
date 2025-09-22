import re
import gc
from typing import Dict, List, Optional

import torch
from transformers import LlamaForCausalLM, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

gc.enable()

class LLMAdapter:
    """Generic adapter class for all LLMs
    """
    def format_prompt(self, utterance: str, state: List[Dict[str, str]]=None, role='user', **kwargs):
        """Given a conversation state and user utterance, format the utterance into a prompt format for the given LLM

        Args:
            state (List[Dict[str, str]]): _description_
            utterance (str): _description_

        Returns:
            List[Dict[str, str]]: completed prompt
        """
        raise NotImplementedError()

    def complete(self, prompt: List[Dict[str, str]], **kwargs):
        """Given a prompt, provide a completion response

        Args:
            prompt (List[Dict[str, str]]): prompt given to the LLM model to generate a response

        Returns:
            Tuple[str, List[Dict[str, str]]]: response string and new state with response
        """
        raise NotImplementedError()


class LLMClient:
    def __init__(self, adapter: LLMAdapter, persona: Optional[str]=None) -> None:
        self.adapter = adapter
        self.state = []
        self.persona = persona

    def state_display(self):
        pass

    def state_show(self):
        pass

    def initiate(self):
        prompt_init = self.persona
        if prompt_init is None:
            prompt_init = 'Say something to start the conversation'

        prompt = [{
            'role': 'system',
            'content': prompt_init
        }]
        response, self.state = self.adapter.complete(prompt)
        return response


    def respond(self, utterance):
        prompt = []
        if self.persona is not None:
            # Add system prompt if available
            prompt = self.adapter.format_prompt(self.persona, state=prompt, role='system')

        # append previous utterances
        prompt += self.state
        # add current utterance
        prompt = self.adapter.format_prompt(utterance, state=prompt)

        # get response
        response, self.state = self.adapter.complete(prompt)
        return response


class  GenericAdapter(LLMAdapter):
    def __init__(self, model_path, quantization=None, **kwargs) -> None:
        self.quantization_options = None
        if quantization == '4bit':
            self.quantization_options = BitsAndBytesConfig(load_in_4bit=True,
                                                        bnb_4bit_quant_type='nf4',
                                                        bnb_4bit_use_double_quant=True,
                                                        bnb_4bit_compute_dtype=torch.bfloat16)
        elif quantization  == '8bit':
            self.quantization_options = BitsAndBytesConfig(load_in_8bit=True)

        self.model = AutoModelForCausalLM.from_pretrained(model_path,
                                                          device_map='auto',
                                                          quantization_config=self.quantization_options)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.generation_config = {
            "max_new_tokens": 256,
            "return_full_text": False,
            "repetition_penalty": 1.1,
            "pad_token_id": self.tokenizer.eos_token_id
        }
        self.pipeline = pipeline("text-generation",
                                 model=self.model,
                                 tokenizer=self.tokenizer,
                                 pad_token_id=self.tokenizer.eos_token_id)

    def format_prompt(self, utterance: str, state: List[Dict[str, str]] = None, role='user', **kwargs):
        prompt = []
        if state is not None:
            prompt = state

        prompt.append({
            'role': role,
            'content': utterance
        })
        return prompt

    def complete(self, prompt: List[Dict[str, str]], **kwargs):
        with torch.no_grad():
            obj_response = self.pipeline(prompt, **self.generation_config)
            response = obj_response[0]['generated_text']
            if type(response) is list:
                response = response[-1]['content']
            new_state = prompt[1:] + self.format_prompt(response, role='assistant')

            gc.collect()
            torch.cuda.empty_cache()


        return response, new_state
