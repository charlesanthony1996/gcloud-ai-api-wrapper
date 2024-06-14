from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain.chains import RetrievalQA

from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

import re

from .globImports import llm_groq, llm_openai, hs_store

# Examples for few-shot learning
explanation_examples = [
    {"question": "Please explain why this is Hate Speech: 'Cultures have never been fair and this is a fact. Muslims for example are just a bunch of misogynist monkeys!", "answer":"This statement would generally be considered hate speech because it's derogatory and targets a specific group of people based on their religion and ethnicity, using offensive language to degrade and dehumanize them."},
    {"question": "Please explain why this is Hate Speech: 'I went to a shit load of therapy. I still don't feel safe in my own house with all these black people around.'", "answer":"That statement would be considered hate speech because it contains racial prejudice and derogatory language towards a specific racial group, implying that their presence is inherently threatening or unsafe."},
    {"question": "Please explain why this is Hate Speech: 'Asians are basically just a bunch of slobs and I don't like how they are accepted here.'", "answer":"This statement is considered hate speech since it makes a derogatory generalization about a specific racial group and expresses dislike or resentment towards their acceptance in a certain community."},
    {"question": "Please explain why this is Hate Speech: 'Freedom of speech is under attack. It's being taken away by Zionist Occupation Government.'", "answer":"That statement is considered hate speech due to it perpetuating a conspiracy theory about a supposed 'Zionist Occupation Government' controlling or suppressing freedom of speech, which can contribute to anti-Semitic sentiments. "},
    {"question": "Please explain why this is Hate Speech: 'Gypsies are literally destroying Europe. They are literally invading and destroying our beautiful continent. They are a cancer on our race and they must be stopped'", "answer":"This statement is considered hate speech because as dehumanizes and vilifies an entire ethnic group, using inflammatory language like 'destroying,' 'invading,' and comparing them to a 'cancer.'"},
    {"question": "Please explain why this is Hate Speech: 'I think the whole equality thing is getting out of hand. We are different, that’s how were made!'", "answer":"This statement is considered hate speech as the idea that men and women are fundamentally different and should therefore have different rights or opportunities is a core belief of sexism."},
    {"question": "Please explain why this is Hate Speech: 'We should isolate gays because they are spreading AIDS all around the world.'", "answer":"This statement is considered hate speech for the reason that it suggests that trans actors should not be allowed to play certain roles in movies, implying that they are not capable or qualified to do so because of their gender identity. "},
    {"question": "Please explain why this is Hate Speech: 'The real problem is that fat people do not want to change'", "answer":"This statement is considered hate speech because it suggests that fat people are lazy or unwilling to change, which is a harmful stereotype that can perpetuate discrimination and stigmatization of individuals who are overweight or obese."},
    {"question": "Please explain why this is Hate Speech: 'Downy children should be isolated, they cannot cope with education in normal schools and are detrimental to their peer's education.'", "answer":"This statement is considered hate speech since it implies that children with Down syndrome are detrimental to their peers' education, which is a baseless and harmful assumption. This contributes to a culture of exclusion and marginalization of individuals with disabilities."},
    {"question": "Please explain why this is Hate Speech: 'Foreigners on UK benefits leaps 41% \in 5 years with over 400K now handed payouts. This nonsense has got to stop'", "answer":"This statement is considered hate speech as it suggests that foreigners or immigrants in the UK are a burden on the welfare system and that they are taking advantage of benefits that should be reserved for UK citizens"},
    {"question": "Please explain why this is Hate Speech: 'You sad fuck. I'm going to restore my edit, and you are going to leave it, or else I will knee you in the groin yourself, prick.'", "answer":"This statement is considered hate speech for the reason that it is using offensive and threatening language to attack and intimidate a specific person based on their perceived identity or behavior."},
]

# Prompt template for explanations
explanation_example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{question}"),
        ("ai", "{answer}")
    ]
)

# Few-shot prompt setup
explanation_few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=explanation_example_prompt,
    examples=explanation_examples
)

# Final prompt template for generating explanations
explanation_final_prompt_explanation = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a model designed to generate explanations for why a comment is hate speech using only the context and examples provided."),
        explanation_few_shot_prompt,
        ("human", "{context}"),
        ("human", "{question}"),
    ]
)

##############################################
# Hate Speech Explanation
##############################################

# Setup for retrieval-based QA with few-shot examples
explanation_qa_with_source_few_shot = RetrievalQA.from_chain_type(
    llm=llm_groq,
    chain_type='stuff',
    retriever = hs_store.as_retriever(),
    chain_type_kwargs={'prompt': explanation_final_prompt_explanation},
)

# Function to get explanation from LLM
def explanation_llm_prompting_calls(message):
    result = explanation_qa_with_source_few_shot.invoke(f"Please explain why this comment can be considered Hate Speech: {message}")        
    result = re.sub('\\\\', '', result["result"])
    result = re.sub('\\n','', result)
    result = re.sub('""', '"', result) # to check for issue of double quotes
    return result