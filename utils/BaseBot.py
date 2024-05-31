# Built-in Imports
import os

# Third-Party Imports
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_pinecone import PineconeVectorStore

# Internal Imports
from utils.document_processor import DocumentProcessor
from utils.embedding_manager import EmbeddingManager
from utils.pinecone_manager import PineconeManager

# require the following ENV variables to be set:
# - OPENAI_API_KEY
# - PINECONE_API_KEY


class BaseChatBot:
    def __init__(
        self,
        filepath: str,
        encoding: str = "utf-8",
        index_name: str = "langchain-demo",
        namespace: str = "default_namespace",
        dimension: int = 768,
        metric: str = "cosine",
        cloud: str = "aws",
        region: str = "us-east-1",
        embed_docs: bool = True,
        chunk_id_label: str = "doc",
        chunk_size: int = 1000,
        chunk_overlap: int = 4,
        inference_model: str = "gpt-3.5-turbo",
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
        self.embed_docs = embed_docs
        self.doc_processor = DocumentProcessor(
            filepath=self.filepath,
            encoding=self.encoding,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.pinecone_manager = PineconeManager(
            api_key=os.getenv("PINECONE_API_KEY"),
            index_name=self.index_name,
            namespace=self.namespace,
            dimension=768,
            metric=self.metric,
            cloud=self.cloud,
            region=self.region,
        )
        self.embedding_manager = EmbeddingManager(id_label=chunk_id_label)
        # self.initialize_pinecone()
        # self.initialize_model()
        # self.create_chain()

    def initialize_pinecone(self, upsert_vectors: bool = True):
        try:
            self.docs = self.doc_processor.load_and_split_documents()

            # Generate embeddings
            if self.embed_docs:
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
