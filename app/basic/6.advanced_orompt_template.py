from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers  import StrOutputParser

model = ChatOllama(model="gemma3:1b")


prompt = PromptTemplate(template = "write a brief report on {topic}",
                        input_variables =['topic'])





prompt2 = PromptTemplate(template = "write 4  points  from {text}  strictly it should be only 4",
                        input_variables = ['text'])

parser = StrOutputParser()

chain =  prompt | model | parser | prompt2 | model | parser


result = chain.invoke({"topic":"Deep learning for 3d object reconstrction"})
print(result)



