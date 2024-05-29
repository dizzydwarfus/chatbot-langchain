from bot.FAQChatBot import FAQChatBot
import streamlit as st

bot = FAQChatBot(filepath="./data/sample_text.txt", encoding="utf-8")

st.set_page_config(page_title="Random Financial Analyst Bot", page_icon=":robot:")
with st.sidebar:
    st.title("Random Financial Analyst Bot")


def generate_response(input):
    result = bot.rag_chain.invoke(input)
    return result


if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Welcome, let's make you some money!"}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if input := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": input})
    with st.chat_message("user"):
        st.write(input)

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Getting your answer from the investment vault..."):
            response = generate_response(input)
            st.write(response)
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)
