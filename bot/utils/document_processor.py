from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from bot.utils._logger import MyLogger

logger = MyLogger(
    name="document_processor.py",
    log_file="C:/Users/lianz/Python/chatbot-langchain/chatbot.log",
).logger


class DocumentProcessor:
    """Initialize the DocumentProcessor class.

    Args:
        source_name (str, optional): source_name which acts as source of raw_string or text. Defaults to None.
        raw_string (str, optional): Raw text/string file to split into Documents. Defaults to None.
        encoding (str, optional): Encoding system. Defaults to "utf-8".
        chunk_size (int, optional): Document chunk size counted by character count, not token count. Defaults to 1000.
        chunk_overlap (int, optional): Characters overlap between documents. Defaults to 4.
    """

    def __init__(
        self,
        source_name: str,
        raw_string: str,
        encoding: str = "utf-8",
        chunk_size: int = 1000,
        chunk_overlap: int = 4,
    ):
        self.source_name = source_name
        self.raw_string = raw_string
        self.encoding = encoding
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_and_split_documents(self):
        try:
            logger.info("Splitting documents...")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
            )

            docs = [
                Document(page_content=x, metadata={"source": self.source_name})
                for x in text_splitter.split_text(self.raw_string)
            ]

            logger.info(f"Loaded and split {len(docs)} documents.")

            return docs

        except Exception as e:
            logger.error(
                f"Error loading and splitting documents: {type(e).__name__}: {e}"
            )
            raise e
