from typing import List, Dict, Tuple

class TODTaskFormatter:
    @staticmethod
    def format_prompt(candidates: List[str], descriptions: Dict[str, str], utterance: str):
        """ Given the similarity scores for candidate classes, format prompt to
            pass to LLM for candidate selection

        Args:
            candidates (List[str]): list of candidates intents, index
                                    corresponds to sims index
            descriptions (Dict[str, str]): textual descriptions corresponding to
                                           each intent class
            utterance (str): original utterance for model to evaluate
        """
        output_text = '\n'.join([
            f"Given the user said \"{utterance}\"\nPlease give the 'intent' that best reflect what the user is saying/asking for, based on which of the following intents has a description best matching the user's utterance:",
            "",
            *[f"intent: {intent}\ndescription: {descriptions[intent]}\n" for intent in candidates],
            "",
            "Please give the intent name only, do not provide reasoning.",
            "The intent is: "
        ])
        return output_text

    @staticmethod
    def format_output(output: Tuple):
        """ Given a model prompt, output and list of target intents, parse the
            model's prediction

        Args:
            output: Tuple: should be in the form of (str, List[str]) with the
                first element being the output text
        """
        return output[0]


class TODCoTTaskFormatter:
    @staticmethod
    def format_prompt(candidates: List[str], descriptions: Dict[str, str], utterance: str):
        """ Given the similarity scores for candidate classes, format prompt to
            pass to LLM for candidate selection

        Args:
            candidates (List[str]): list of candidates intents, index
                                    corresponds to sims index
            descriptions (Dict[str, str]): textual descriptions corresponding to
                                           each intent class
            utterance (str): original utterance for model to evaluate
        """
        output_text = '\n'.join([
            f"Given the user said \"{utterance}\"\nPlease give the 'intent' that best reflect what the user is saying/asking for, based on which of the following intents has a description best matching the user's utterance:",
            "",
            *[f"intent: {intent}\ndescription: {descriptions[intent]}\n" for intent in candidates],
            "",
            "Please provide your reasoning for your choice of intent, and then give the intent by saying 'the intent is: $INTENT', replacing $INTENT with the intent label.",
        ])
        return output_text

    @staticmethod
    def format_output(output: Tuple):
        """ Given a model prompt, output and list of target intents, parse the
            model's prediction

        Args:
            output: Tuple: should be in the form of (str, List[str]) with the
                first element being the output text
        """
        return output[0]

