from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, \
    TrainingArguments
from datasets import load_dataset


class FineTunedGPT2Chatbot:
    def __init__(self):
        self.model, self.tokenizer = self.load_model_fine_tuned_gpt2()

    def fine_tune_gpt2(self):
        # Load pre-trained GPT-2 model and tokenizer
        model_name = "gpt2"
        model = GPT2LMHeadModel.from_pretrained(model_name)
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)

        # Load the Wikitext dataset
        wikitext_dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
        train_text = "\n".join(wikitext_dataset["train"]["text"])

        # Save the Wikitext dataset to a file
        dataset_path = "wikitext_dataset.txt"
        with open(dataset_path, "w", encoding="utf-8") as file:
            file.write(train_text)

        # Load the dataset from the saved file
        train_dataset = TextDataset(
            tokenizer=tokenizer,
            file_path=dataset_path,
            block_size=128,  # Adjust block size as needed
        )

        # Prepare the Trainer
        training_args = TrainingArguments(
            output_dir="./gpt2-fine-tuned",
            overwrite_output_dir=True,
            num_train_epochs=3,  # Adjust epochs as needed
            per_device_train_batch_size=4,  # Adjust batch size as needed
            save_steps=10_000,
            save_total_limit=2,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False,
            ),
            train_dataset=train_dataset,
        )

        # Fine-tune the model
        trainer.train()

        model.save_pretrained("../gpt2-fine-tuned")
        tokenizer.save_pretrained("../gpt2-fine-tuned")

    def load_model_fine_tuned_gpt2(self):
        model_name = "gpt2-fine-tuned"  # Use the path to your fine-tuned model directory
        model = GPT2LMHeadModel.from_pretrained(model_name)
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        return model, tokenizer

    def generate_response(self, user_input, max_length=500):
        input_ids = self.tokenizer.encode(user_input, return_tensors="pt")

        # Generate a response using the fine-tuned model
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
