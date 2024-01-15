from transformers import DistilBertForQuestionAnswering, DistilBertTokenizer
import torch

class DistilBERTChatbot:
    def __init__(self):
        self.model, self.tokenizer = self.load_model_distilbert_qa()

    def load_model_distilbert_qa(self):
        model_name = "distilbert-base-cased-distilled-squad"
        model = DistilBertForQuestionAnswering.from_pretrained(model_name)
        tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        return model, tokenizer

    def generate_response(self, user_input, max_length=128):
        inputs = self.tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
        outputs = self.model(**inputs)
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits
        start_index = torch.argmax(start_logits, dim=1).item()
        end_index = torch.argmax(end_logits, dim=1).item()
        answer_span = self.tokenizer.decode(inputs["input_ids"][0][start_index:end_index + 1], skip_special_tokens=True)
        return answer_span