import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from main import ChatBot


class ChatSimulator:
    def __init__(self):
        self.bot = ChatBot(filepath="./data/sample_text.txt", encoding="utf-8")
        self.messages = [
            {"role": "assistant", "content": "Welcome, let's make you some money!"}
        ]

    def generate_response(self, input):
        result = self.bot.rag_chain.invoke(input)
        return result

    def display_messages(self):
        for message in self.messages:
            print(f"{message['role'].capitalize()}: {message['content']}")

    def add_user_message(self, user_input):
        self.messages.append({"role": "user", "content": user_input})
        print(f"User: {user_input}")

    def add_assistant_message(self, response):
        self.messages.append({"role": "assistant", "content": response})
        print(f"Assistant: {response}")

    def run_chat_simulation(self):
        self.display_messages()

        while True:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]:
                break
            self.add_user_message(user_input)

            print("Getting your answer from the investment vault...")
            response = self.generate_response(user_input)
            self.add_assistant_message(response)

            self.display_messages()


if __name__ == "__main__":
    simulator = ChatSimulator()
    simulator.run_chat_simulation()
