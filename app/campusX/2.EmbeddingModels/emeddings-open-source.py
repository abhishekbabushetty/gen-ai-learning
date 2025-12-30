from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")



text = ["delhi is the capital of india",
        "mumbai is the capital of maharashtra",
        "kolkata is the capital of west bengal",
        "chennai is the capital of tamil nadu",
        "hyderabad is the capital of telangana"]

ans = embedding.embed_documents(text)



print(str(ans))