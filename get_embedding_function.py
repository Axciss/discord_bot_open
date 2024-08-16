from langchain_huggingface import HuggingFaceEmbeddings

def get_embedding_function(model_name):
    # initialize Hugging Face embedding model
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    return embeddings