from langchain.schema.runnable import Runnable
from langchain.schema.runnable import RunnableConfig


class LoggingRunnable(Runnable):
    def __init__(self, logger):
        self.logger = logger

    def invoke(self, input, config: RunnableConfig = None):
        self.logger.info(f"Input to model: {input}")
        return input
