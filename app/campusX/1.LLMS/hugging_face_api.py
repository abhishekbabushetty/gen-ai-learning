from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation",
    max_new_tokens=200,
    temperature=0.7,
)

chat = ChatHuggingFace(llm=llm)

resp = chat.invoke("Write a poem about cricket")
print(resp.content)
