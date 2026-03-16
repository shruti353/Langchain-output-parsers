from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
import os

load_dotenv()

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant"
)

# Define schema using Pydantic
class Facts(BaseModel):
    fact1: str = Field(description="fact 1 about the topic")
    fact2: str = Field(description="fact 2 about the topic")
    fact3: str = Field(description="fact 3 about the topic")

parser = JsonOutputParser(pydantic_object=Facts)

template = PromptTemplate(
    template="Give 3 facts about {topic}\n{format_instruction}",
    input_variables=['topic'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

chain = template | llm | parser

result = chain.invoke({'topic': 'mythology'})
print(result)