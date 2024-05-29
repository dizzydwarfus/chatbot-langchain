from utils.BaseBot import BaseChatBot


class FAQChatBot(BaseChatBot):
    def get_prompt_template(self):
        return """
        You are a helpful assistant that answers frequently asked questions. The user will ask you questions, and you will use the provided context to answer. 
        If you don't know the answer, just say you don't know. 
        Be concise and clear.

        Context: {context}
        Question: {question}
        Answer: 
        """
