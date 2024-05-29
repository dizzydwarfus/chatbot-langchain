from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_pinecone import PineconeVectorStore
import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import numpy as np


class ChatBot:
    def __init__(
        self,
        filepath: str,
        encoding: str = "utf-8",
        index_name: str = "langchain-demo",
        namespace: str = "default_namespace",
    ):
        load_dotenv()
        self.filepath = os.path.abspath(filepath)
        self.encoding = encoding
        self.index_name = index_name
        self.namespace = namespace
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

            if self.index_name not in self.pc.list_indexes().names():
                self.pc.create_index(
                    name=self.index_name,
                    dimension=768,  # Dimension of the embeddings based on embedding model
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1",
                    ),
                )

            self.index = self.pc.Index(self.index_name)

            # Function to generate unique IDs for each document chunk
            def generate_document_id(base_id, chunk_index):
                return f"{base_id}_chunk{chunk_index}"

            # Prepare data for upsert
            vectors_to_upsert = []
            for i, doc in enumerate(self.docs):
                doc_id = generate_document_id("kpi_doc", i)
                embedding = self.embeddings.embed_documents(doc.page_content)
                flatten_embedding = np.array(embedding).flatten().tolist()
                vectors_to_upsert.append(
                    {
                        "id": doc_id,
                        "values": flatten_embedding,
                        "metadata": {
                            "text": doc.page_content,
                            "source": doc.metadata["source"],
                        },
                    }
                )

            # Upsert data into Pinecone
            self.index.upsert(
                vectors=vectors_to_upsert,
                namespace=self.namespace,
            )

            self.docsearch = PineconeVectorStore.from_existing_index(
                index_name=self.index_name,
                embedding=self.embeddings,
                namespace=self.namespace,
            )
        except Exception as e:
            print(f"Error initializing Pinecone: {e}")
            raise

    def initialize_model(self):
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
