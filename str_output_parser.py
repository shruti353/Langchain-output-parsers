from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
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

prompt1 = template1.invoke({"topic": "black hole"})

result1 = llm.invoke(prompt1)

prompt2 = template2.invoke({"text": result1.content})

result2 = llm.invoke(prompt2)

print(result2.content)