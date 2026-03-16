from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant"
)

# Prompt 1 → detailed report
template1 = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=["topic"]
)

# Prompt 2 → summary
template2 = PromptTemplate(
    template="Write a 5 line summary of the following text:\n{text}",
    input_variables=["text"]
)

parser = StrOutputParser()

chain = template1 | llm | parser | template2 | llm | parser

result=chain.invoke({'topic': 'Generative AI'})
print(result)