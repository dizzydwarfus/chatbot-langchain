from bot.utils.BaseBot import BaseChatBot


class FAQChatBot(BaseChatBot):
    def get_prompt_template(self):
        # return """
        # You are a helpful assistant that answers frequently asked questions. The user will ask you questions, and you will use the provided context to answer.
        # If you don't know the answer, just say you don't know.

        # Previous Messages: {messages}
        # Context: {context}
        # Question: {question}
        # Answer:
        # """
        return """
        You are a helpful assistant that answers frequently asked questions. The user will ask you questions, and you will use the provided context to answer. 
        If you don't know the answer, just say you don't know.
        You will be provided with all the exchange and history between the user and the assistant. Please respond to the last user's request or question.

        Context: {context}
        Question: {question}
        Answer: 
        """
