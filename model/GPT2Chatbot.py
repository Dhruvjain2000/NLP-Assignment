from transformers import GPT2LMHeadModel, GPT2Tokenizer


class GPT2Chatbot:
    def __init__(self):
        self.model, self.tokenizer = self.load_model_gpt2()

    def load_model_gpt2(self):
        model_name = "gpt2"
        model = GPT2LMHeadModel.from_pretrained(model_name)
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        return model, tokenizer

    def generate_response(self, user_input, max_length=500):
        input_ids = self.tokenizer.encode(user_input, return_tensors="pt")

        # Generate a response using the model
        output = self.model.generate(
            input_ids,
            max_length=max_length,
            num_beams=5,
            no_repeat_ngram_size=3,
            top_k=20,
            top_p=0.95,
            temperature=0.2,
        )

        # Decode and return the response
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return response
