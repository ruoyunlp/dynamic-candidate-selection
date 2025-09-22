import torch

from transformers import AutoModel, AutoTokenizer

from typing import List


def masked_mean(tensor, mask):
    """ Function for calculating mean of embeddings with masking

    Args:
        tensor (torch.Tensor): batch_size * seq_len * hidden_size
        mask (torch.Tensor): batch_size * seq_len

    Returns:
        _type_: _description_
    """
    hidden_size = tensor.shape[-1]

    # set masked tokens (i.e. [PAD]) to 0
    out = tensor.reshape(-1, hidden_size) * mask.flatten().reshape(-1, 1)

    # reset shape to batch_size x seq_len x hidden_size
    out = out.reshape(tensor.shape)

    # sum and divide to calculate mean
    out = out.sum(dim=1) / mask.sum(dim=1).reshape(-1, 1)
    return out


class BaseInferenceModel:
    def get_sentence_embeds(self, sents: List[str], norm=True) -> torch.Tensor:
        """ Produce contextual representations for given sentences

        Args:
            sents (List[str]): List of sentences to get representations for
            norm (bool, optional): normalise embeddings. Defaults to True.

        Returns:
            torch.Tensor: contextual sentence representations, torch Tensor of shape num_sentences x hidden_size
        """
        raise NotImplementedError()


class GTEModel(BaseInferenceModel):
    def __init__(self, model_type, device) -> None:
        super().__init__()
        self.model = AutoModel.from_pretrained(model_type).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_type)
        self.device = device

    def get_sentence_embeds(self, sents: List[str], norm=True) -> torch.Tensor:
        encodings = self.tokenizer(sents, padding=True, truncation=True, return_tensors='pt')
        encodings = encodings.to(self.device)
        input_ids = encodings.input_ids
        attention_mask = encodings.attention_mask

        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = out.last_hidden_state.detach()

        # omit [MASK] token embeddings from averaging
        attention_mask = attention_mask & (input_ids != self.tokenizer.mask_token_id)
        embeds = masked_mean(last_hidden_state, attention_mask).cpu()

        if norm:
            embeds = embeds / embeds.norm(dim=1).reshape(-1, 1)

        return embeds
