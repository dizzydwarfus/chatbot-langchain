from langchain_openai import OpenAIEmbeddings
import os
from bot.utils._logger import MyLogger

logger = MyLogger(
    name="embedding_manager.py",
    log_file="./chatbot.log",
).logger


class EmbeddingManager:
    def __init__(self, id_label: str = "doc"):
        self.embeddings = OpenAIEmbeddings(
            api_key=os.getenv("OPENAI_API_KEY"), model="text-embedding-3-small"
        )
        self.id_label = id_label

    def generate_embeddings(self, docs):
        try:
            logger.info("Generating embeddings...")
            vectors = []
            for i, doc in enumerate(docs):
                embedding = self.embeddings.embed_query(doc.page_content)
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
            logger.info(f"Generated {len(vectors)} embeddings.")
            return vectors
        except Exception as e:
            logger.error(f"Error generating embeddings: {type(e).__name__}: {e}")
            raise e
