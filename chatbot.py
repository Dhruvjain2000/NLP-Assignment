import os
import urllib.request

import requests
import torch
from bs4 import BeautifulSoup
import re
import spacy
import nltk
from torch.nn.utils.rnn import pad_sequence

from openai import OpenAI
from sumy.nlp.stemmers import Stemmer
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lsa import LsaSummarizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer, BertForSequenceClassification, BertTokenizer, \
    BertForQuestionAnswering, DistilBertTokenizer, DistilBertForQuestionAnswering, get_linear_schedule_with_warmup, \
    AdamW

from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, \
    TrainingArguments

from datasets import load_dataset

from torch.utils.data import Dataset

# from chatterbot import ChatBot
# from chatterbot.trainers import ListTrainer
# from chatterbot.trainers import ChatterBotCorpusTrainer

# Load the spaCy English model
nlp = spacy.load("en_core_web_lg")
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# Set your OpenAI API key
os.environ.setdefault("OPENAI_API_KEY", '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')


# Model 1
def ask_chatgpt(prompt):
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="text-davinci-002",
    )

    data = chat_completion.choices[0]['text']
    return data


# Model 2
def get_weather_info(city):
    api_key = '2a01d8a4ac8d58d66f3c9da814a1b954'
    base_url = f'http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}'
    try:
        response = requests.get(base_url)
        data = response.json()

        if data['cod'] == '404':
            return "City not found. Please provide a valid city name."

        temperature = data['main']['temp']
        description = data['weather'][0]['description']
        return f"The weather in {city} is {description} with a temperature of {temperature}°C."
    except Exception as e:
        return f"An error occurred while fetching weather information: {str(e)}"


def handle_greetings(question):
    greetings = ["hi", "hello", "hey"]
    normalized_question = question.lower().split()  # Split the question into words

    for greeting in greetings:
        if greeting in normalized_question:
            return "Hello! How can I help you today?"

    return None


def handle_math_calculation(question):
    if any(char.isdigit() for char in question):
        try:
            result = eval(question)
            return f"The result of {question} is {result}"
        except:
            return "Unable to perform the mathematical calculation."
    else:
        return None


def get_wikipedia_content(url):
    response = urllib.request.urlopen(url)
    soup = BeautifulSoup(response, 'html.parser')
    paragraphs = soup.find_all('p')
    content = ' '.join([paragraph.text for paragraph in paragraphs])
    return content


def preprocess_text(text):
    # Basic text preprocessing to remove special characters and multiple spaces
    text = re.sub(r'\[\d+\]', '', text)  # Remove numeric citations like [1], [2], etc.
    return text


def get_city_country_info(city):
    username = 'dhruv2000'
    base_url = f'http://api.geonames.org/searchJSON?q={city}&maxRows=1&username={username}'

    try:
        response = requests.get(base_url)
        data = response.json()

        if 'geonames' not in data or len(data['geonames']) == 0:
            return "City not found. Please provide a valid city name."

        city_info = data['geonames'][0]
        city_name = city_info.get('name', '')
        country_name = city_info.get('countryName', '')

        return f"The city of {city_name} is located in the country of {country_name}."
    except Exception as e:
        return f"An error occurred while fetching city and country information: {str(e)}"


def qa_bot(user_question, training_data_url):
    # Check for greetings
    greeting_response = handle_greetings(user_question)
    if greeting_response:
        return greeting_response

    # Check for math calculations
    math_response = handle_math_calculation(user_question)
    if math_response:
        return math_response

    main_keyword = get_all_important_words(user_question.lower())
    # Check for weather-related queries
    if 'weather' in user_question.lower():
        city_match = None if main_keyword == "weather" else main_keyword

        if city_match:
            return get_weather_info(city_match)
        else:
            return "Please provide a city name to check the weather."

    if main_keyword:
        main_keyword_url = f'https://en.wikipedia.org/wiki/{main_keyword}'
        training_data = get_wikipedia_content(main_keyword_url)
        preprocessed_training_data = preprocess_text(training_data)

        if main_keyword.split('_')[0].lower() in preprocessed_training_data.lower():
            relevant_info = extract_relevant_info(user_question, preprocessed_training_data)
            return relevant_info
        else:
            return "I'm sorry, I couldn't find information related to your question."
    else:
        return "No relevant keyword found in the question."


def clean_and_summarize(text, max_sentences=5, max_characters=500, min_characters=300):
    parser = PlaintextParser.from_string(text[0: 700], Tokenizer("english"))
    stemmer = Stemmer("english")
    summarizer = LsaSummarizer(stemmer)
    summary = summarizer(parser.document, max_sentences)
    summary_text = "_".join(str(sentence) for sentence in summary)

    if len(summary_text) > max_characters:
        sentences = [str(sentence) for sentence in summary[:max_sentences]]
        last_sentence = ""
        for sentence in sentences:
            last_sentence += sentence + " "
            if len(last_sentence) >= min_characters:
                break

        summary_text = last_sentence

    return summary_text


def extract_relevant_info(user_question, training_data):
    cleaned_and_summarized_info = clean_and_summarize(training_data)
    if len(training_data) < 100:
        return "You need to provide more context. This can lead to several possible answers depending upon the context"

    return cleaned_and_summarized_info


def get_all_important_words(text):
    words = nltk.word_tokenize(text)
    tags = nltk.pos_tag(words)

    relevant_words = []

    for i in range(len(tags)):
        word, tag = tags[i]
        if tag in ['NN', 'NNS', 'JJ']:  # Considering nouns and adjectives as important
            relevant_words.append(word)

    if relevant_words:
        return "_".join(relevant_words)
    else:
        return None


def extract_entities(query):
    doc = nlp(query)

    if doc.ents:
        keywords = [ent.text for ent in doc.ents]
        print("The keywords are: " + str(keywords))

        main_keyword = keywords[0]
    else:
        main_keyword = get_all_important_words(query)

        if main_keyword is None:
            return "No entities found in the query."

    wikipedia_url = f'https://en.wikipedia.org/wiki/{main_keyword}'
    return wikipedia_url


# Model 3
def load_model_gpt2():
    model_name = "gpt2"
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    return model, tokenizer


def generate_response_gpt2(model, tokenizer, user_input, max_length=500):
    input_ids = tokenizer.encode(user_input, return_tensors="pt")

    # Generate a response using the model
    output = model.generate(
        input_ids,
        max_length=max_length,
        num_beams=5,
        no_repeat_ngram_size=3,
        top_k=20,  # Adjust this value
        top_p=0.95,
        temperature=0.2,  # Adjust this value
    )

    # Decode and return the response
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response


# Model 4
class QADataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'input_ids': item['input_ids'],
            'attention_mask': item['attention_mask'],
            'start_positions': item['start_positions'],
            'end_positions': item['end_positions']
        }


def load_model_distilbert_qa():
    model_name = "distilbert-base-cased-distilled-squad"
    model = DistilBertForQuestionAnswering.from_pretrained(model_name)
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    return model, tokenizer


def generate_response_distilbert_qa(model, tokenizer, user_input, max_length=128):
    # Tokenize input and get input IDs
    inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)

    # Make a forward pass to get start and end logits
    outputs = model(**inputs)
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits

    # Print the logits for debugging
    print("Start logits:", start_logits)
    print("End logits:", end_logits)

    # Get the most likely start and end positions along the sequence dimension
    start_index = torch.argmax(start_logits, dim=1).item()
    end_index = torch.argmax(end_logits, dim=1).item()

    # Get the answer span from the original text
    answer_span = tokenizer.decode(inputs["input_ids"][0][start_index:end_index + 1], skip_special_tokens=True)

    return answer_span


def your_data_processing_function(text):
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-cased-distilled-squad")
    examples = []

    # Split text into paragraphs
    paragraphs = text.split('\n')

    for paragraph in tqdm(paragraphs, desc="Tokenizing paragraphs"):
        # Tokenize the paragraph
        tokens = tokenizer.encode_plus(paragraph, return_tensors="pt", padding=True, truncation=True)
        input_ids = tokens['input_ids'].squeeze(0)
        attention_mask = tokens['attention_mask'].squeeze(0)

        # Placeholder: Add logic to get start and end positions (e.g., for QA task)
        # In this example, set start and end positions to the first and last tokens
        start_positions = torch.tensor([1])  # Replace with your logic
        end_positions = torch.tensor([len(input_ids) - 1])  # Replace with your logic

        # Add the processed example to the list
        examples.append({
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'start_positions': start_positions,
            'end_positions': end_positions
        })

    return examples


def load_and_preprocess_wikitext():
    # Load Wikitext dataset
    wikitext_dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    train_text = "\n".join(wikitext_dataset["train"]["text"])
    # Process your data to get input_ids, attention_mask, start_positions, end_positions
    processed_data = your_data_processing_function(train_text)

    return processed_data
    return train_text


def preprocess_qa_data(dataset):
    batch_size = 8
    return DataLoader(QADataset(dataset), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)


def collate_fn(batch):
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-cased-distilled-squad")

    # Extract input_ids, attention_mask, start_positions, and end_positions from the batch
    input_ids_list = [str(item['input_ids']) for item in batch]  # Replace 'input_ids' with the correct key
    attention_mask_list = [item['attention_mask'] for item in batch]
    start_positions = torch.stack([item['start_positions'] for item in batch])
    end_positions = torch.stack([item['end_positions'] for item in batch])

    # Tokenize the inputs
    inputs = tokenizer(input_ids_list, return_tensors="pt", padding=True, truncation=True)

    # Add the start_positions and end_positions to the inputs
    inputs['start_positions'] = start_positions
    inputs['end_positions'] = end_positions

    return inputs




def fine_tune_bert(save_model_path='./fine_tuned_model', epochs=3):
    # Placeholder: Replace with your actual data loading and preprocessing logic for Wikitext
    wikitext_dataset = load_and_preprocess_wikitext()

    # Placeholder: Preprocess your QA dataset
    train_dataloader = preprocess_qa_data(wikitext_dataset)

    model, tokenizer = load_model_distilbert_qa()
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                                num_training_steps=len(train_dataloader) * epochs)

    # Fine-tuning loop
    for epoch in range(epochs):
        model.train()
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}"):
            optimizer.zero_grad()
            input_ids, attention_mask, start_positions, end_positions = batch
            outputs = model(input_ids, attention_mask=attention_mask, start_positions=start_positions,
                            end_positions=end_positions)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()

        # Save the model at the end of each epoch
        model.save_pretrained(save_model_path)
        tokenizer.save_pretrained(save_model_path)

    print(f"Model has been fine-tuned and saved to {save_model_path}")


# Model 5 Fine tune GPT 2
def fine_tune_gpt2():
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

    model.save_pretrained("./gpt2-fine-tuned")
    tokenizer.save_pretrained("./gpt2-fine-tuned")


def load_model_fine_tuned_gpt2():
    model_name = "./gpt2-fine-tuned"  # Use the path to your fine-tuned model directory
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    return model, tokenizer


def generate_response_fine_tuned_gpt2(model, tokenizer, user_input, max_length=500):
    input_ids = tokenizer.encode(user_input, return_tensors="pt")

    # Generate a response using the fine-tuned model
    output = model.generate(
        input_ids,
        max_length=max_length,
        num_beams=5,
        no_repeat_ngram_size=3,
        top_k=20,
        top_p=0.95,
        temperature=0.2,
    )

    # Decode and return the response
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response


# Model for ChatterBot
# def load_model_chatterbot():
#     chatbot = ChatBot('ChatBot')
#     trainer_corpus = ChatterBotCorpusTrainer(chatbot)
#     trainer_corpus.train('chatterbot.corpus.english')
#     return chatbot
#
# def generate_response_chatterbot(chatbot, user_input):
#     response = chatbot.get_response(user_input)
#     return str(response)

# fine_tune_gpt2()
fine_tune_bert()
