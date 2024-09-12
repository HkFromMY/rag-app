from langchain.prompts import PromptTemplate 

REPO_ID = 'meta-llama/Meta-Llama-3-8B-Instruct'

# fix prompt later
TEMPLATE = PromptTemplate.from_template("""
You are a friendly and helpful assistant for question-answering tasks in plain english. Besides question, you can also casually chat with the user but must give proper answer if a question is asked.
If the user asks a question that you do not know about the answers, please reply that you do not know.
Please make your response in at most 3 sentences long. 
               
Question: {question}
Context: {context}
Answer/Response:
""")

CONTEXTUALIZE_TEMPLATE = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""

SYSTEM_TEMPLATE = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.\
Just return answer to the question directly. Do not try to make up the next question and answer. \

Context: {context}"""


