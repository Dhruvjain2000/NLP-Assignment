import re
import requests
import urllib.request
import nltk
from bs4 import BeautifulSoup
from nltk.stem import StemmerI
from sumy.nlp.stemmers import Stemmer
from sumy.nlp.tokenizers import Tokenizer

from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lsa import LsaSummarizer
import spacy


# Load the spaCy English model
nlp = spacy.load("en_core_web_lg")
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

class CustomModel:
    def handle_greetings(self, question):
        greetings = ["hi", "hello", "hey"]
        normalized_question = question.lower().split()  # Split the question into words

        for greeting in greetings:
            if greeting in normalized_question:
                return "Hello! How can I help you today?"

        return None

    def handle_math_calculation(self, question):
        if any(char.isdigit() for char in question):
            try:
                result = eval(question)
                return f"The result of {question} is {result}"
            except:
                return "Unable to perform the mathematical calculation."
        else:
            return None

    def get_wikipedia_content(self, url):
        response = urllib.request.urlopen(url)
        soup = BeautifulSoup(response, 'html.parser')
        paragraphs = soup.find_all('p')
        content = ' '.join([paragraph.text for paragraph in paragraphs])
        return content

    def preprocess_text(self, text):
        # Basic text preprocessing to remove special characters and multiple spaces
        text = re.sub(r'\[\d+\]', '', text)  # Remove numeric citations like [1], [2], etc.
        return text

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

    def get_city_country_info(self, city):
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

    def qa_bot(self, user_question, training_data_url):
        # Check for greetings
        greeting_response = self.handle_greetings(user_question)
        if greeting_response:
            return greeting_response

        # Check for math calculations
        math_response = self.handle_math_calculation(user_question)
        if math_response:
            return math_response

        main_keyword = self.get_all_important_words(user_question.lower())
        # Check for weather-related queries
        if 'weather' in user_question.lower():
            city_match = None if main_keyword == "weather" else main_keyword

            if city_match:
                return self.get_weather_info(city_match)
            else:
                return "Please provide a city name to check the weather."

        if main_keyword:
            main_keyword_url = f'https://en.wikipedia.org/wiki/{main_keyword}'
            training_data = self.get_wikipedia_content(main_keyword_url)
            preprocessed_training_data = self.preprocess_text(training_data)

            if main_keyword.split('_')[0].lower() in preprocessed_training_data.lower():
                relevant_info = self.extract_relevant_info(user_question, preprocessed_training_data)
                return relevant_info
            else:
                return "I'm sorry, I couldn't find information related to your question."
        else:
            return "No relevant keyword found in the question."

    def clean_and_summarize(self, text, max_sentences=5, max_characters=500, min_characters=300):
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

    def extract_relevant_info(self, user_question, training_data):
        cleaned_and_summarized_info = self.clean_and_summarize(training_data)
        if len(training_data) < 100:
            return "You need to provide more context. This can lead to several possible answers depending upon the context"

        return cleaned_and_summarized_info

    def get_all_important_words(self, text):
        words = nltk.word_tokenize(text)
        tags = nltk.pos_tag(words)

        relevant_words = []

        for i in range(len(tags)):
            word, tag = tags[i]
            if tag in ['NN', 'NNS', 'JJ']:  # Considering nouns and adjectives as important
                relevant_words.append(word)

        if relevant_words:
            return "_".join(relevant_words)
        else :
            return None

    def extract_entities(self, query):
        doc = nlp(query)

        if doc.ents:
            keywords = [ent.text for ent in doc.ents]
            print("The keywords are: " + str(keywords))

            main_keyword = keywords[0]
        else:
            main_keyword = self.get_all_important_words(query)

            if main_keyword is None:
                return "No entities found in the query."

        wikipedia_url = f'https://en.wikipedia.org/wiki/{main_keyword}'
        return wikipedia_url

    def handle_weather_info(self, user_question):
        words = nltk.word_tokenize(user_question.lower())
        if 'weather' in words:
            # Assume the city name is mentioned after "weather"
            index_of_weather = words.index('weather')
            if index_of_weather < len(words) - 1:
                city = words[index_of_weather + 1]
                return self.get_weather_info(city)
            else:
                return "Please provide a city name to check the weather."
        else:
            return None

    def get_all_important_nouns(self, text):
        words = nltk.word_tokenize(text)
        tags = nltk.pos_tag(words)
        chunks = nltk.ne_chunk(tags)

        relevant_nouns = [chunk.leaves()[0][0] for chunk in chunks if
                          isinstance(chunk, nltk.Tree) and chunk.label() == 'GPE']

        if relevant_nouns:
            return " ".join(relevant_nouns)
        else:
            return None

    def handle_wikipedia_search(self, user_question):
        main_keyword = self.get_all_important_nouns(user_question.lower())
        if main_keyword:
            main_keyword_url = f'https://en.wikipedia.org/wiki/{main_keyword}'
            training_data = self.get_wikipedia_content(main_keyword_url)
            preprocessed_training_data = self.preprocess_text(training_data)

            if main_keyword.split('_')[0].lower() in preprocessed_training_data.lower():
                relevant_info = self.extract_relevant_info(user_question, preprocessed_training_data)
                return relevant_info
            else:
                return "I'm sorry, I couldn't find information related to your question."
        else:
            return "No relevant keyword found in the question."

    def handle_google_search(self, user_question):
        # Placeholder: Implement Google search functionality using web scraping
        # You can use libraries like BeautifulSoup and requests for web scraping
        return "Google search functionality not implemented yet."

    def handle_custom_function(self, user_question):
        # Placeholder: Implement your custom functionality based on user_question
        return "Custom functionality not implemented yet."

    def handle_query(self, user_question):
        # You can add more conditions or modify the order based on your requirements
        weather_response = self.handle_weather_info(user_question)
        if weather_response:
            return weather_response

        wikipedia_response = self.handle_wikipedia_search(user_question)
        if wikipedia_response:
            return wikipedia_response

        google_response = self.handle_google_search(user_question)
        if google_response:
            return google_response

        custom_function_response = self.handle_custom_function(user_question)
        if custom_function_response:
            return custom_function_response

        return "Sorry, I couldn't find information related to your question."



