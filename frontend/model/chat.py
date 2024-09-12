from langchain_community.llms import HuggingFaceEndpoint 
from langchain.chains import ConversationalRetrievalChain, create_history_aware_retriever, create_retrieval_chain
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from utils.constant import REPO_ID, TEMPLATE, CONTEXTUALIZE_TEMPLATE, SYSTEM_TEMPLATE
from model.rag import create_retriever
import streamlit as st
import time

def get_session_history(session_id):
    store = st.session_state.chat_history
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

@st.cache_resource
def load_model():

    model = HuggingFaceEndpoint(
        repo_id=REPO_ID,
        temperature=0.1,
        model_kwargs={ 'max_length': 1024 },
    )

    return model

@st.cache_resource
def create_chain():
    model = load_model()
    retriever = create_retriever()

    contextualize_prompt = ChatPromptTemplate.from_messages(
        [
            ('system', CONTEXTUALIZE_TEMPLATE),
            MessagesPlaceholder('chat_history'),
            ('human', '{input}')
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm=model,
        retriever=retriever,
        prompt=contextualize_prompt,
    )

    chat_prompt = ChatPromptTemplate.from_messages(
        [
            ('system', SYSTEM_TEMPLATE),
            MessagesPlaceholder('chat_history'),
            ('human', '{input}')
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm=model, prompt=chat_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key='input',
        history_messages_key='chat_history',
        output_messages_key='answer',
    )

    return chain

def send_message(message):
    chain = create_chain()
    response = chain.invoke(
        { 'input': message },
        { 'configurable': { 'session_id': '1' } }
    )
    message = response['answer']

    for chunk in message.split():
        if '<|eot_id|>' in chunk:
            yield chunk.replace('<|eot_id|>', '')
        else:
            yield chunk + " "

        time.sleep(0.05)
