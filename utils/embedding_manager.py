from langchain.embeddings import HuggingFaceEmbeddings
import numpy as np
from langchain_openai import OpenAIEmbeddings
import os


class EmbeddingManager:
    def __init__(self, id_label: str = "doc"):
        # self.embeddings = OpenAIEmbeddings(
        #     api_key=os.getenv("OPENAI_API_KEY"), model="text-embedding-3-small"
        # )
        self.embeddings = HuggingFaceEmbeddings()
        self.id_label = id_label

    def generate_embeddings(self, docs):
        vectors = []
        for i, doc in enumerate(docs):
            embedding = self.embeddings.embed_documents(doc.page_content)
            # flatten_embedding = np.array(embedding).flatten().tolist()
            vectors.append(
                {
                    "id": f"{self.id_label}_chunk{i}",
                    "values": embedding,
                    "metadata": {
                        "text": doc.page_content,
                        "source": doc.metadata["source"],
                    },
                }
            )
        return vectors