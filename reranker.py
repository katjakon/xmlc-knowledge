
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class BGEReranker:

    def __init__(self, model_path, device=None):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if device is not None:
            self.model.to(device)

    def similarities(self, input_ids, attention_masks=None):
        """
        input_ids: torch.Tensor of shape (batch_size, seq_len)
        attention_masks: torch.Tensor of shape (batch_size, seq_len)

        Returns:
        torch.Tensor of shape (batch_size, )
        """
        self.model.eval()
        with torch.no_grad():
            scores = self.model(
                input_ids=input_ids, 
                attention_mask=attention_masks,
                return_dict=True).logits.view(-1, ).float()
        return scores
    
    def tokenize(self, pair, max_len=128):
        return self.tokenizer(pair, padding=True, truncation=True, return_tensors='pt', max_length=max_len)