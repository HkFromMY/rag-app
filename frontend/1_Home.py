import streamlit as st 

from model.chat import send_message

st.title('RAG Application')

# ensure that messages field is inside
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = {}

# render all messages stored previously
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# main body
prompt = st.chat_input('What is up?')
if prompt:
    with st.chat_message('user'):
        st.markdown(prompt)

    st.session_state.messages.append({ 'role': 'user', 'content': prompt })

    with st.chat_message('assistant'):

        message_generator = send_message(prompt)
        response = st.write_stream(message_generator)

    st.session_state.messages.append({ 'role': 'assistant', 'content': response }) 