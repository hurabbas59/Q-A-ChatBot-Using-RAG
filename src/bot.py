# from dotenv import load_dotenv
import os
import openai
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from dotenv import loadenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize FAISS and Embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002",api_key=OPENAI_API_KEY)
vector_store = FAISS.load_local(
    "faiss_store", embeddings, allow_dangerous_deserialization=True
)

PROMPT = """\
You are an AI assistant. Your role is to provide accurate and concise answers based on questions.
Use the following context to answer the user's question. 
if there are calculations, perform it accurately and provide accurate answers.

Question: {question}

Relevant Information: {context}

be specific to your provided data and do not reply anything except the provided information.

"""

def get_docs(query, top_k=5):
    results = vector_store.similarity_search(
        query=query,
        k=top_k,
    )
    formated_docs = ""
    
    for result in results:
        formated_docs += result.page_content + "\n"
    
    return formated_docs

def run(query):
    docs = get_docs(query)

    prompt = PromptTemplate(
        input_variables=["question", "context"], template=PROMPT
    )
    llm = ChatOpenAI(model="gpt-4o-mini",api_key=OPENAI_API_KEY)
    chain = prompt | llm | StrOutputParser()
    data = {
        "question": query,
        "context": docs
    }

    response = chain.invoke(data)
    
    return response
