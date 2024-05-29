from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader


class DocumentProcessor:
    def __init__(self, filepath, encoding="utf-8", chunk_size=1000, chunk_overlap=4):
        self.filepath = filepath
        self.encoding = encoding
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_and_split_documents(self):
        loader = TextLoader(self.filepath, encoding=self.encoding)
        documents = loader.load()
        text_splitter = CharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        docs = text_splitter.split_documents(documents)
        return docs
