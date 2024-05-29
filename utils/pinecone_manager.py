from pinecone import Pinecone, ServerlessSpec
from typing import Literal


class PineconeManager:
    def __init__(
        self,
        api_key: str,
        index_name: str,
        namespace: str,
        dimension: int,
        metric: str = Literal["cosine", "dotproduct", "euclidean"],
        cloud: str = "aws",
        region: str = "us-east-1",
    ):
        self.api_key = api_key
        self.index_name = index_name
        self.namespace = namespace
        self.dimension = dimension
        self.metric = metric
        self.cloud = cloud
        self.region = region
        self.pc = None
        self.index = None

    def initialize(self):
        self.pc = Pinecone(api_key=self.api_key)
        if self.index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric=self.metric,
                spec=ServerlessSpec(cloud=self.cloud, region=self.region),
            )
        self.index = self.pc.Index(self.index_name)

    def upsert_vectors(self, vectors, batch_size=2):
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i : i + batch_size]
            self.index.upsert(vectors=batch, namespace=self.namespace)
