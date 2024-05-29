from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings

# from langchain.llms import HuggingFaceHub
from langchain_openai import ChatOpenAI
from langchain import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_pinecone import PineconeVectorStore
import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec


class ChatBot:
    def __init__(self, filepath, encoding="utf-8"):
        load_dotenv()
        self.filepath = os.path.abspath(filepath)
        self.encoding = encoding
        self.initialize_pinecone()
        self.initialize_model()
        self.create_chain()

    def initialize_pinecone(self):
        try:
            # Create a TextLoader instance with the given file path and encoding
            self.loader = TextLoader(self.filepath, encoding=self.encoding)
            self.documents = self.loader.load()

            # Split documents into chunks, ensuring they don't exceed the specified size
            self.text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=4)
            self.docs = self.text_splitter.split_documents(self.documents)

            self.embeddings = HuggingFaceEmbeddings()

            # Initialize Pinecone client
            self.pc = Pinecone(
                api_key=os.getenv("PINECONE_API_KEY"),
            )

            self.index_name = "langchain-demo"
            self.namespace = "press-release"

            if self.index_name not in self.pc.list_indexes().names():
                self.pc.create_index(
                    name=self.index_name,
                    dimension=768,  # Adjust this dimension based on your embedding model
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",  # Replace with your cloud provider
                        region="us-east-1",  # Replace with your region
                    ),
                )
                self.docsearch = PineconeVectorStore.from_documents(
                    documents=self.docs,
                    embedding=self.embeddings,
                    index_name=self.index_name,
                    namespace=self.namespace,
                )
            else:
                self.docsearch = PineconeVectorStore.from_existing_index(
                    embedding=self.embeddings,
                    index_name=self.index_name,
                    namespace=self.namespace,
                )
        except Exception as e:
            print(f"Error initializing Pinecone: {e}")
            raise

    def initialize_model(self):
        # repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
        # self.llm = HuggingFaceHub(
        #     repo_id=repo_id,
        #     model_kwargs={"temperature": 0.8, "top_k": 50},
        #     huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_KEY"),
        # )
        self.llm = ChatOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name="gpt-3.5-turbo",
            temperature=0,
        )

    def create_chain(self):
        template = """
        You are a financial analyst that serves individual retail investors. The user will ask you questions about a stock they are interested in. 
        Use the following piece of context to answer the question. 
        If you don't know the answer, just say you don't know. 
        Be concise.

        Context: {context}
        Question: {question}
        Answer: 
        """
        prompt = PromptTemplate(
            template=template, input_variables=["context", "question"]
        )
        self.rag_chain = (
            {
                "context": self.docsearch.as_retriever(),
                "question": RunnablePassthrough(),
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )


# Example usage in app.py
# if __name__ == "__main__":
# Assuming this script is named `main.py`
# bot = ChatBot(filepath="./data/sample_text.txt", encoding="utf-8")
