# Built-in Imports
import os

# Third-Party Imports
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_pinecone import PineconeVectorStore

# Internal Imports
from bot.utils.document_processor import DocumentProcessor
from bot.utils.embedding_manager import EmbeddingManager
from bot.utils.pinecone_manager import PineconeManager

# require the following ENV variables to be set:
# - OPENAI_API_KEY
# - PINECONE_API_KEY

from bot.utils._logger import MyLogger

logger = MyLogger(
    name="BaseChatBot.py",
    log_file="./chatbot.log",
).logger


class BaseChatBot:
    def __init__(
        self,
        source_name: str,
        raw_data: str,
        encoding: str = "utf-8",
        index_name: str = "langchain-demo",
        namespace: str = "default_namespace",
        dimension: int = 768,
        metric: str = "cosine",
        cloud: str = "aws",
        region: str = "us-east-1",
        chunk_id_label: str = "doc",
        chunk_size: int = 1000,
        chunk_overlap: int = 4,
        inference_model: str = "gpt-3.5-turbo",
    ):
        load_dotenv()
        self.source_name = source_name
        self.raw_data = raw_data
        self.encoding = encoding
        self.index_name = index_name
        self.namespace = namespace
        self.dimension = dimension
        self.metric = metric
        self.cloud = cloud
        self.region = region
        self.inference_model = inference_model
        self.doc_processor = DocumentProcessor(
            source_name=self.source_name,
            raw_string=self.raw_data,
            encoding=self.encoding,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.pinecone_manager = PineconeManager(
            api_key=os.getenv("PINECONE_API_KEY"),
            index_name=self.index_name,
            namespace=self.namespace,
            dimension=self.dimension,
            metric=self.metric,
            cloud=self.cloud,
            region=self.region,
        )
        self.embedding_manager = EmbeddingManager(id_label=chunk_id_label)

    def initialize_pinecone(self, upsert_vectors: bool = True, embed_docs: bool = True):
        try:
            logger.info(
                f"Initializing Pinecone... Parameters:\n - Upsert vectors: {upsert_vectors}\n - Embed docs: {embed_docs}"
            )
            self.docs = self.doc_processor.load_and_split_documents()

            # Generate embeddings
            if embed_docs:
                self.vectors_to_upsert = self.embedding_manager.generate_embeddings(
                    self.docs
                )

            # Initialize Pinecone and upsert vectors
            self.pinecone_manager.initialize()

            if upsert_vectors:
                self.pinecone_manager.upsert_vectors(self.vectors_to_upsert)

            # Create PineconeVectorStore from the existing index
            self.docsearch = PineconeVectorStore.from_existing_index(
                index_name=self.index_name,
                embedding=self.embedding_manager.embeddings,
                namespace=self.namespace,
            )
        except Exception as e:
            logger.error(f"Error initializing Pinecone: {type(e).__name__}: {str(e)}")
            raise

    def initialize_model(self):
        try:
            self.llm = ChatOpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                model_name=self.inference_model,
                temperature=0,
            )
        except Exception as e:
            logger.error(
                f"Error initializing OpenAI model: {type(e).__name__}: {str(e)}"
            )
            raise e

    def create_chain(self):
        try:
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
        except Exception as e:
            logger.error(f"Error creating chain: {type(e).__name__}: {str(e)}")
            raise e

    def get_prompt_template(self):
        raise NotImplementedError(
            "Subclasses should implement this method to provide specific prompt templates."
        )
