import re

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain.chains import RetrievalQA

from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

from .globImports import llm_groq, llm_openai

import requests
##############################################
# Counter Speech Generation
##############################################

cs_examples = [
    {"input": "This is Hate Speech: 'All Muslims should be deported, no one wants them here.' Please generate counter speech for it of the following type: Facts", "output": "Then why did we ask them to come in the first place, wasn't it top save our NHS, run the trains and post office. Non of those were doing very well last time I looked, I think we might actually still bneed them.", "cs_type": 'Facts'},
    {"input": "This is Hate Speech: 'Muslims had destroyed a Christian statue in Italy. Italy should ask itself why they are allowed to build mosques and why is tolerating these despicable people.' Please generate counter speech for it of the following type: Hypocrisy", "output": "Why have we built Churches all over the world? How many holy buildings have we demolished over history? This is never acceptable, but can we really be casting the first stone?", "cs_type": 'Hypocrisy'},
    {"input": "This is Hate Speech: 'We need to Burn the Koran and all the Mosques down - this will solve all problems in this country.' Please generate counter speech for it of the following type: Consequences", "output": "Sounds very much like incitement to me, do be careful.", "cs_type": 'Consequences'},
    {"input": "This is Hate Speech: 'Another rapist from Pakistan. We should deport them all back to their disgusting country.' Please generate counter speech for it of the following type: Affiliation", "output": "All the people from Pakistan that I know are really nice! No need to generalize too much!", "cs_type": 'Affiliation'},
    {"input": "This is Hate Speech: 'What is the point in listening to Muslim women? They are all retarded.' Please generate counter speech for it of the following type: Denouncing", "output": "You decide whether or not people are intelligent based on their religion? Well, this is not a way forward.", "cs_type": 'Denouncing'},
    {"input": "This is Hate Speech: 'Liberal double standards! Muslims oppress women, homosexuals and other cultures but are not held responsible due to our society's tolerance! The government should stop submitting to these people.' Please generate counter speech for it of the following type: Humor", "output": "Nah mate, they really don't.", "cs_type": 'Humor'},
    {"input": "This is Hate Speech: 'Islam is not a peaceful religion, rapes on children made by Muslims are copious, 10 people in a night, 100 in a week. In the meanwhile, the media does not speak about these facts.' Please generate counter speech for it of the following type: Positive", "output": "The media is not ignoring those rape crimes. Those who commit the crimes are extremists. The majority of Muslims are peaceful and respect people. It is better we think about how we can prevent the crimes.", "cs_type": 'Positive'},
]

cs_example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
        ("ai", "{cs_type}" )
    ]
)

cs_few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=cs_example_prompt,
    examples=cs_examples
)

cs_final_prompt = ChatPromptTemplate.from_messages(
    [
        #in the following JSON-format: {{'Answer': '', 'CS_type': ''}}, 
        ("system", "You are a model designed to generate counter speech using only the context and examples provided. Your tone should be: conversational, spartan tone and less corporate jargon."),
        cs_few_shot_prompt,
        ("human", "{context}"),
        ("human", "{question}"),
    ])

cs_loader = TextLoader('./files/cs_types.md')
cs_data = cs_loader.load()
cs_types = cs_data[0].page_content

cs_headers_to_split_on = [
    ('#', "Header 1")
]

cs_text_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=cs_headers_to_split_on
)

cs_types_split = cs_text_splitter.split_text(cs_types)

embeddings = OpenAIEmbeddings()

cs_store = Chroma.from_documents(
    cs_types_split,
    embeddings,
    collection_name='cs_types',
    persist_directory = 'counter_speech_types_store',
)

cs_store.persist()

cs_qa_with_source_few_shot = RetrievalQA.from_chain_type(
    llm=llm_groq,
    chain_type = 'stuff',
    retriever = cs_store.as_retriever(),
    chain_type_kwargs={'prompt': cs_final_prompt},
    return_source_documents = True
)

def cs_llm_prompting_calls(message, cs_type):
    #result = qa_with_source_few_shot.invoke(f"This is a hateful Comment: '{hate_speech_comment}' because: {explanation}. Please generate counter speech of CS_Type: 'Positive'. Format the response in the following JSON-format: {{'Counterspeech Type': '', 'Answer': ''}}")
    result = cs_qa_with_source_few_shot.invoke(f"This is a hateful comment: {message}. Please generate a counter speech response of type {cs_type}: 'Positive'. Format the response in the following JSON-Format: {{\"Answer\": \"\"}}")
    result = re.sub('\\\\', '', result["result"])
    result = re.sub('\\n','', result)
    result = re.sub('""', '"', result) # to check for issue of double quotes
    return result