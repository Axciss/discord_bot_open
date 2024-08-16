from langchain_huggingface import HuggingFaceEmbeddings

def get_embedding_function(model_name):
    # 初始化 Hugging Face 嵌入模型
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    # def embedding_function(texts):
    #     # 使用 Hugging Face 嵌入模型嵌入文本
    #     return embeddings.embed_documents(texts)

    return embeddings