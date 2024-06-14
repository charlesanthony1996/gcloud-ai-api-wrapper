import os
from openai import OpenAI
from groq import Groq
from dotenv import load_dotenv

import re

from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

from .globImports import llm_groq, llm_openai, hs_store

# Examples for few-shot learning
hs_examples = [
    {"question": "Is this Hate Speech: 'Cultures have never been fair and this is a fact. Muslims for example are just a bunch of misogynist monkeys!' Answer a. Yes or b. No, Answer:", "explanation":"This statement would generally be considered hate speech. It's derogatory and targets a specific group of people based on their religion and ethnicity, using offensive language to degrade and dehumanize them.", "output":" The answer is a. Yes", "hs_type": "islamophobia"},
    {"question": "Is this Hate Speech: 'You deserve way more subscribers. The s*** you say is f***ing gold.' Answer a. Yes or b. No, Answer:", "explanation":"This statement is not hate speech. It's a compliment expressing admiration for someone's content and suggesting they deserve more recognition or subscribers.", "output":" The answer is b. No", "hs_type": "None"},
    {"question": "Is this Hate Speech: 'I went to a shit load of therapy. I still don't feel safe in my own house with all these black people around.' Answer a. Yes or b. No, Answer:", "explanation":"That statement would be considered hate speech. It contains racial prejudice and derogatory language towards a specific racial group, implying that their presence is inherently threatening or unsafe.", "output":" The answer is a. Yes", "hs_type": "anti-black racism"},
    {"question": "Is this Hate Speech: 'Because people becomes evil easily, it is in our genes' . Answer a. Yes or b. No, Answer:", "explanation":" This statement does not constitute hate speech on its own. It seems to express a belief about human behavior and genetics. ", "output":" The answer is b. No", "hs_type": "None"},
    {"question": "Is this Hate Speech: 'Asians are basically just a bunch of slobs and I don't like how they are accepted here.' Answer a. Yes or b. No, Answer:", "explanation":"This statement is considered hate speech. It makes a derogatory generalization about a specific racial group and expresses dislike or resentment towards their acceptance in a certain community.", "output":" The answer is a. Yes", "hs_type": "anti-asian racism"},
    {"question": "Is this Hate Speech: 'Freedom of speech is under attack. It's being taken away by Zionist Occupation Government.' Answer a. Yes or b. No, Answer:", "explanation":"That statement is considered hate speech. It perpetuates a conspiracy theory about a supposed 'Zionist Occupation Government' controlling or suppressing freedom of speech, which can contribute to anti-Semitic sentiments. ", "output":" The answer is a. Yes", "hs_type": "antisemitism"},
    {"question": "Is this Hate Speech: 'Gypsies are literally destroying Europe. They are literally invading and destroying our beautiful continent. They are a cancer on our race and they must be stopped' Answer a. Yes or b. No, Answer:", "explanation":"This statement is considered hate speech. It dehumanizes and vilifies an entire ethnic group, using inflammatory language like 'destroying,' 'invading,' and comparing them to a 'cancer.'", "output":" The answer is a. Yes", "hs_type": "anti-gypsyism"},
    {"question": "Is this Hate Speech: 'I think the whole equality thing is getting out of hand. We are different, that’s how were made!' Answer a. Yes or b. No, Answer:", "explanation":"This statement is considered hate speech. Yes, this statement can be considered sexist. The idea that men and women are fundamentally different and should therefore have different rights or opportunities is a core belief of sexism.", "output":" The answer is a. Yes", "hs_type": "sexism"},
    {"question": "Is this Hate Speech: 'We should isolate gays because they are spreading AIDS all around the world.' Answer a. Yes or b. No, Answer:", "explanation":"This statement is considered hate speech. It suggests that trans actors should not be allowed to play certain roles in movies, implying that they are not capable or qualified to do so because of their gender identity. ", "output":" The answer is a. Yes", "hs_type": "trans-hostility"},
    {"question": "Is this Hate Speech: 'The real problem is that fat people do not want to change' Answer a. Yes or b. No, Answer:", "explanation":"This statement is considered hate speech. It suggests that fat people are lazy or unwilling to change, which is a harmful stereotype that can perpetuate discrimination and stigmatization of individuals who are overweight or obese.", "output":" The answer is a. Yes", "hs_type": "fatphobia"},
    {"question": "Is this Hate Speech: 'Downy children should be isolated, they cannot cope with education in normal schools and are detrimental to their peer's education.' Answer a. Yes or b. No, Answer:", "explanation":"This statement is considered hate speech. It implies that children with Down syndrome are detrimental to their peers' education, which is a baseless and harmful assumption. This contributes to a culture of exclusion and marginalization of individuals with disabilities.", "output":" The answer is a. Yes", "hs_type": "ableism"},
    {"question": "Is this Hate Speech: 'Foreigners on UK benefits leaps 41% in 5 years with over 400K now handed payouts. This nonsense has got to stop' Answer a. Yes or b. No, Answer:", "explanation":"This statement is considered hate speech. It suggests that foreigners or immigrants in the UK are a burden on the welfare system and that they are taking advantage of benefits that should be reserved for UK citizens", "output":" The answer is a. Yes", "hs_type": "hostility towards migrants"},
    {"question": "Is this Hate Speech: 'You sad fuck. I'm going to restore my edit, and you are going to leave it, or else I will knee you in the groin yourself, prick.' Answer a. Yes or b. No, Answer:", "explanation":"This statement is considered hate speech. It is using offensive and threatening language to attack and intimidate a specific person based on their perceived identity or behavior.", "output":" The answer is a. Yes", "hs_type": "cyberbullying"},
]

# Prompt template for examples
hs_example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{question}"),
        ("ai", "{output}"),
        ("ai", "{explanation}"),
        ("ai", "{hs_type}"),
    ]
)

# Few-shot prompt setup
hs_few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=hs_example_prompt,
    examples=hs_examples
)

# Final prompt template for hate speech detection
hs_final_prompt = ChatPromptTemplate.from_messages(
[
    ("system", "You are a model designed to detect Hate Speech using only the context provided."),
    hs_few_shot_prompt,
    ("human", "{context}"),
    ("human", "{question}"),
]
)

##############################################
# Hate Speech Detection
##############################################

# Setup for retrieval-based QA with few-shot examples
qa_with_source_few_shot = RetrievalQA.from_chain_type(
llm=llm_openai,
chain_type='stuff',
retriever = hs_store.as_retriever(),
chain_type_kwargs={'prompt': hs_final_prompt},
return_source_documents = True,
)

# Function to detect hate speech
def llm_prompting_calls(message):
    #result_groq = qa_with_source_few_shot.invoke(f"Do you think this comment is hate speech? Comment: {message}. Please answer a. Yes or b. No. Don't provide an explanation and format the response in the following JSON-Format: {{\"Answer\": \"\",  \"hs_type\": \"\"}}")        
    result = qa_with_source_few_shot.invoke(f"Do you think this comment is hate speech? comment: {message} Please also answer a. Yes or b. No in the following JSON-format: {{'Answer': '', 'hs_type': ''}} ")    
    result = re.sub('\\\\', '', result["result"])
    result = re.sub('\\n','', result)
    result = re.sub('\\n','', result)
    result = re.sub('""', '"', result) # to check for issue of double quotes
    return result






