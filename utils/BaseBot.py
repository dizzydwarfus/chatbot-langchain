from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_pinecone import PineconeVectorStore
import os

from document_processor import DocumentProcessor
from embedding_manager import EmbeddingManager
from pinecone_manager import PineconeManager

# require the following ENV variables to be set:
# - OPENAI_API_KEY
# - PINECONE_API_KEY


class BaseChatBot:
    def __init__(
        self,
        filepath,
        encoding="utf-8",
        index_name="langchain-demo",
        namespace="default_namespace",
        dimension=768,
        metric="cosine",
        cloud="aws",
        region="us-east-1",
        inference_model="gpt-3.5-turbo",
    ):
        load_dotenv()
        self.filepath = os.path.abspath(filepath)
        self.encoding = encoding
        self.index_name = index_name
        self.namespace = namespace
        self.dimension = dimension
        self.metric = metric
        self.cloud = cloud
        self.region = region
        self.inference_model = inference_model
        self.pinecone_manager = PineconeManager(
            api_key=os.getenv("PINECONE_API_KEY"),
            index_name=self.index_name,
            namespace=self.namespace,
            dimension=768,
            metric=self.metric,
            cloud=self.cloud,
            region=self.region,
        )
        self.embedding_manager = EmbeddingManager()
        self.initialize_pinecone()
        self.initialize_model()
        self.create_chain()

    def initialize_pinecone(self):
        try:
            # Load and split documents
            doc_processor = DocumentProcessor(
                filepath=self.filepath, encoding=self.encoding
            )
            docs = doc_processor.load_and_split_documents()

            # Generate embeddings
            vectors_to_upsert = self.embedding_manager.generate_embeddings(docs)

            # Initialize Pinecone and upsert vectors
            self.pinecone_manager.initialize()
            self.pinecone_manager.upsert_vectors(vectors_to_upsert)

            # Create PineconeVectorStore from the existing index
            self.docsearch = PineconeVectorStore.from_existing_index(
                index_name=self.index_name,
                embedding=self.embedding_manager.embeddings,
                namespace=self.namespace,
            )
        except Exception as e:
            print(f"Error initializing Pinecone: {type(e).__name__}: {str(e)}")
            raise

    def initialize_model(self):
        self.llm = ChatOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name=self.inference_model,
            temperature=0,
        )

    def create_chain(self):
        template = self.get_prompt_template()
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

    def get_prompt_template(self):
        raise NotImplementedError(
            "Subclasses should implement this method to provide specific prompt templates."
        )
