import os, sys
from tkinter import *

from chat import Chat


dir = os.path.dirname(os.path.abspath(__file__))
jsonFile = os.path.join(dir, "../", 'data/intent')
chat = Chat(jsonPath="data/intents.json", train=True)

def process_input(text):
    tag = chat.predict_class(text)
    return chat.answer(text, tag)

def main():
    print("Welcome to the terminal app. Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Exiting the application. Goodbye!")
            break
        response = process_input(user_input)
        print(f"Bot: {response}")

if __name__ == "__main__":
    main()