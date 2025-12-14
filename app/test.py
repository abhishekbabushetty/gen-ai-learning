from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from pydantic import BaseModel, Field
from typing import Literal,Optional
from dotenv import load_dotenv


load_dotenv()

class Answer(BaseModel):
    agent_name : str = Field(description = "your name")
    agent_domain : str = Field(description = "your area") 
    response: Optional[str] = Field(description = "keep the response very very short") 

model = ChatOllama(model = "gemma3:1b")
#model = ChatGoogleGenerativeAI(model = "gemini-2.5-flash" )
model = model.with_structured_output(Answer)

prompt_template = PromptTemplate(
    [
        ("system","your name is {name} and your {domain} expert"),
        ("human" , "introduce yourself explain about {topic}")
    ]
)


prompt = prompt_template.invoke({
    "name":"Abhishek b shetty",
    "domain":"AI",
    "topic" : "Supervised learning in dl"
})


answer = model.invoke(prompt)


print(answer)



