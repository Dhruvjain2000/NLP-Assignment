from flask import Flask, render_template, request, jsonify
from chatbot import extract_entities, qa_bot, ask_chatgpt, load_model_gpt2, generate_response_gpt2, \
    load_model_distilbert_qa, \
    generate_response_distilbert_qa, load_model_fine_tuned_gpt2
from model.ChatGPT import ChatGPT
from model.CustomModel import CustomModel
from model.DistilBERTChatbot import DistilBERTChatbot
from model.FineTunedGPT2Chatbot import FineTunedGPT2Chatbot
from model.GPT2Chatbot import GPT2Chatbot

# load_model_chatterbot, generate_response_chatterbot

app = Flask(__name__)

distilbert_chatbot = DistilBERTChatbot()
gpt2_chatbot = GPT2Chatbot()
fine_tuned_gpt2_chatbot = FineTunedGPT2Chatbot()
chatgpt = ChatGPT(api_key='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')
customModel = CustomModel()

@app.route('/')
def hello_world():
    return render_template('index.html')


@app.route("/ask", methods=['POST'])
def ask():
    try:
        # Retrieve the message from the form data
        message = str(request.form['messageText'])

        selected_model = str(request.form['modelSelection'])

        # Call the qa_bot function with the user's question and the Wikipedia URL
        if selected_model == "custom":
            # Call the extract_entities function to get the Wikipedia URL
            wikipedia_url = extract_entities(message.lower())
            if wikipedia_url != "No entities found in the query.":
                # response = qa_bot(message, wikipedia_url)
                response = customModel.qa_bot(message, wikipedia_url)
            else:
                response = jsonify({'answer': "No entities found in the query."})

        elif selected_model == "gpt2":
            response = gpt2_chatbot.generate_response(message)
            # model_gpt2, tokenizer_gpt2 = load_model_gpt2()
            # response = generate_response_gpt2(model_gpt2, tokenizer_gpt2, message)

        elif selected_model == "chatgpt":
            response = chatgpt.ask_chatgpt(message)

        elif selected_model == "bert":
            # model_bert, tokenizer_bert = load_model_distilbert_qa()
            # response = generate_response_distilbert_qa(model_bert, tokenizer_bert, message)
            response = distilbert_chatbot.generate_response(message)

        elif selected_model == "fine-tuned-gpt2":
            # model_gpt2, tokenizer_gpt2 = load_model_fine_tuned_gpt2()
            # response = generate_response_gpt2(model_gpt2, tokenizer_gpt2, message)
            response = fine_tuned_gpt2_chatbot.generate_response(message)

        # elif selected_model == "chatterbot":
        #     chatbot = load_model_chatterbot()
        #     response = generate_response_chatterbot(chatbot, message)

        # Print the message for debugging purposes
        print(f"Received message: {message}")

        # Return the response as JSON
        return jsonify({'answer': response})

    except Exception as e:
        # Handle exceptions gracefully
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run()
