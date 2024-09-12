from langchain_community.vectorstores import Chroma 
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings

import os

embedding_model = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')

def create_retriever():
    chroma_db = Chroma(persist_directory=f'{os.getcwd()}\\model\\db', embedding_function=embedding_model)
    retriever = chroma_db.as_retriever(search_type="mmr", search_kwargs={'k': 5, 'fetch_k': 20 })

    return retriever