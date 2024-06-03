from bot.FAQChatBot import FAQChatBot
import streamlit as st
from io import StringIO
from bot.utils._logger import MyLogger

#####################################################

# Set page config

#####################################################
if "state" not in st.session_state:
    st.session_state["state"] = st.set_page_config(
        page_title="Random ChatBot",
        page_icon=":robot:",
        layout="wide",
        initial_sidebar_state="auto",
    )

if "bot" not in st.session_state:
    st.session_state["bot"] = FAQChatBot(
        source_name=None,
        raw_data=None,
        encoding=None,
        index_name=None,
        namespace=None,
        dimension=None,
        chunk_id_label=None,
        chunk_size=None,
        chunk_overlap=None,
        inference_model=None,
    )

logger = MyLogger(name="app.py", log_file="./chatbot.log").logger


with st.sidebar:
    st.title("Configuration")

    col1, col2 = st.columns(2)
    input_type = st.selectbox("Input Type", ["Text", "File"])

    source_name = col1.text_input("Source Name", placeholder="Source name...")
    encoding = col2.text_input("Encoding", "utf-8")

    if input_type == "Text":
        data = st.text_area(
            "Data", placeholder="Copy and paste your context data here..."
        )
        # if data:
        string_rep = StringIO(data)

    elif input_type == "File":
        file_upload = st.file_uploader("Upload FAQ File", type=["md", "txt"])

        if file_upload:
            st.write(file_upload)
            string_rep = StringIO(file_upload.getvalue().decode("utf-8"))

    col3, col4 = st.columns(2)

    index_name = col3.text_input(
        "Index Name", "openai-embedder", help="Index name in Vector DB"
    )
    namespace = col4.text_input(
        "Namespace", "doc", help="Namespace for the embeddings in Vector DB"
    )
    dimension = st.number_input(
        "Dimension", value=1536, help="Dimension of the embedding model"
    )
    chunk_id_label = st.text_input(
        "Chunk ID Label", "doc", help="Chunk ID Label for the embeddings"
    )

    col5, col6 = st.columns(2)

    chunk_size = col5.number_input(
        "Chunk Size", value=1000, help="Chunk size for the embeddings"
    )
    chunk_overlap = col6.number_input(
        "Chunk Overlap", value=4, help="Chunk overlap for the embeddings"
    )
    inference_model = st.selectbox(
        "Inference Model",
        ["gpt-3.5-turbo", "gpt-4-turbo"],
        help="Inference model for the chatbot",
    )

    regen_embeddings = st.checkbox("Regenerate embeddings", key="regenerate_embeddings")
    upsert_vectors = st.checkbox("Upsert vectors", key="upsert_vectors")

    if st.button("Initialize ChatBot"):
        st.info("Initializing ChatBot...", icon="⏳")

        logger.info(
            f"Initializing ChatBot with the following parameters:\nSource Name: {source_name}\nEncoding: {encoding}\nIndex name: {index_name}\nNamespace: {namespace}\nDimension: {dimension}\nChunk ID Label: {chunk_id_label}\nChunk Size: {chunk_size}\nChunk Overlap: {chunk_overlap}\nInference Model: {inference_model}"
        )
        st.session_state.bot = FAQChatBot(
            source_name=source_name,
            raw_data=string_rep.getvalue(),
            encoding=encoding,
            index_name=index_name,
            namespace=namespace,
            dimension=dimension,
            chunk_id_label=chunk_id_label,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            inference_model=inference_model,
        )
        st.session_state.bot.initialize_pinecone(
            upsert_vectors=upsert_vectors, embed_docs=regen_embeddings
        )
        st.session_state.bot.initialize_model()
        st.session_state.bot.create_chain()

        # st.write(st.session_state.bot.docs)


def generate_response(input):
    result = st.session_state.bot.rag_chain.invoke(input)
    return result


st.title("Chat with the Random ChatBot")
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Welcome, how can I help you today?"}
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
        with st.spinner("Getting your answer from the data vault..."):
            response = generate_response(input)
            # response = "This is a random response"
            st.write(response)
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)

# logger.info(f"Session State:\n{json.dumps(dict(st.session_state), indent=2)}")

if st.button("Clear Chat"):
    st.session_state.messages = [
        {"role": "assistant", "content": "Welcome, how can I help you today?"}
    ]

if "bot" in st.session_state:
    st.write("Session State:", st.session_state.bot.namespace)
