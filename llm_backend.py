from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import os
from openai import OpenAI
from groq import Groq
from dotenv import load_dotenv

import json5
import re

# from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain.chains import RetrievalQA

from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_groq import ChatGroq

import llmModules
from llmModules.globImports import llm_groq, llm_openai

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# load config
with open('config.json') as config_file:
    config = json.load(config_file)

# env variables
openai_key = os.getenv('OPENAI_API_KEY')
groq_key = os.getenv('GROQ_API_KEY')
# chosenLLM = config['GENERATING_CS_LLM']
# For using Local Ollama in config change it to this
# "GENERATING_CS_LLM": "localOllama"



if config['llm_type'] == 'openAI':
    client = llmModules.OpenAIClient(api_key=os.getenv('OPENAI_API_KEY'))
elif config['llm_type'] == 'groq':
    client = llmModules.GroqClient(api_key=os.getenv('GROQ_API_KEY'))
else:
    raise ValueError("Unsupported LLM type specified in config.")

# Route to check if filtered comment is HS
@app.route('/api/analyze_hate_speech', methods=['POST'])
def analyze_hate_speech():
    try:
        data = request.json
        user_message = data.get('text', '')
        llm_response = llmModules.llm_prompting_calls(user_message)
        llm_response = json5.loads(llm_response)   

        return {"llm_result": llm_response}, 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500    

# Route to explain why comment is marked as HS first by filter and then LLM (red marking)
@app.route('/api/explain_hate_speech', methods=['POST'])
def explain_hate_speech():
    try:
        data = request.json
        user_message = data.get('text', '')
        explanation_result = llmModules.explanation_llm_prompting_calls(user_message)
        
        return jsonify({"explanation_result": explanation_result}), 200

    except Exception as e:  
        return jsonify({"error": str(e)}), 700

# Route to generate counter speech
@app.route('/api/generate_counter_speech', methods=['POST'])
def generate_counter_speech():
    try:
        print("Generation Request Received")
        data = request.json
        user_message = data.get('text', '')
        cs_type = data.get('cs_type', '')
        print(cs_type, user_message)
        counter_speech_result = llmModules.cs_llm_prompting_calls(user_message, cs_type)
        counter_speech_result = json5.loads(counter_speech_result)
        print("counter speech:", counter_speech_result)
        return {"counter_speech_result": counter_speech_result}, 200
        
        # if chosenLLM == "openaiORgroq":
        
        #     counter_speech_result = llmModules.cs_llm_prompting_calls(user_message, cs_type)
        #     counter_speech_result = json5.loads(counter_speech_result)
        #     print("counter speech:", counter_speech_result)
        #     return {"counter_speech_result": counter_speech_result}, 200
        # else:
        #     # Send the data to the other file's endpoint
        #     url = "http://localollama:6002/api/genOllama"  # Change the URL if needed
        #     payload = {
        #         "message": user_message,
        #         "cs_type": cs_type
        #     }
        #     response = requests.post(url, json=payload)
            
        #     if response.status_code == 200:
        #         counter_speech_result = response.json()
        #     else:
        #         counter_speech_result = f"Error: {response.status_code}, {response.text}"
            
        #     print("Ollama response:", counter_speech_result)
        #     return jsonify({"counter_speech_result": counter_speech_result}), 200

    except Exception as e:  
        return jsonify({"error": str(e)}), 700

@app.route('/')
def home():
    return jsonify({"message": "AI_API-Wrapper Service Running"})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(debug=True, host='0.0.0.0', port=port)


# @app.route('/api/analyze_hate_speech', methods=['POST'])
# def analyze_hate_speech():
#     try:
#         data = request.json
#         user_message = data.get('text', '')
#         response_pre = client.create_completion(user_message)
#         analysis_result_pre = response_pre.choices[0].message.content.strip()
#         print("analysis result: ", analysis_result_pre)
#         if(analysis_result_pre == 'Yes'):
#             response = client.create_completionCS(analysis_result_pre)
#             analysis_result = response.choices[0].message.content.strip()
#         else:
#             analysis_result = analysis_result_pre
#         return jsonify({"analysis_result": analysis_result}), 200
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500