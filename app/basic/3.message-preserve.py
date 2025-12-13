from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.chat import MessagesPlaceholder
from langchain_ollama import ChatOllama

chat_templte = ChatPromptTemplate([
    "system","you are a helper full customer agent",
    MessagesPlaceholder(variable_name="chat_history"),
    ('human','{query}')
])

model =  ChatOllama(model = "gemma3:1b")
chat_history = []

with open('chat_template.txt') as file:
    chat_history.extend(file.readlines())

prompt = chat_templte.invoke({
    'chat_history':chat_history,
    'query':'what is best phone under 10k'

})

answer = model.invoke(prompt)
print(answer.content)
