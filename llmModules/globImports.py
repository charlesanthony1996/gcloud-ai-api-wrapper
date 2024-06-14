from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

import os
from openai import OpenAI
from groq import Groq
from dotenv import load_dotenv

import json5
import re

from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain.chains import RetrievalQA

from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPEN_API_KEY')
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')


llm_groq = ChatGroq(
temperature=1,
model_name = "mixtral-8x7b-32768",
#model_kwargs={"top_p":0.35}
)

llm_openai = ChatOpenAI(
    temperature=0, 
    #model='gpt-4-turbo-preview').bind(
    model='gpt-3.5-turbo')#.bind(
        #response_format={ "type": "json_object" }
        #)
        
        
# Loading hs_store

loader = TextLoader("./files/Definitions_of_Different_Forms_of_Hate_Speech.md")
data = loader.load()
definitions_markdown = data[0].page_content

headers_to_split_on = [
("#", "Header 1"),
("##", "Header 2"),
]

text_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on
)

data = text_splitter.split_text(definitions_markdown)

embeddings = OpenAIEmbeddings()

hs_store = Chroma.from_documents(
data,
embeddings,
collection_name='hate_speech_definitions',
persist_directory = 'hate_sepaker_definitions_store',
)

hs_store.persist()