"""
This gradio demo is used to test the streaming text generation of the model


As the it will produce the output in real time, the model will be able to generate the text in real time.

We will be using gemma-2 model for this.
"""

import time

import gradio as gr
from llama_index.llms.ollama import Ollama

GEMMA_2_B = Ollama(model="gemma:2b-instruct", request_timeout=30.0)

with gr.Blocks(
    theme="soft",
) as demo:
    chatbot = gr.Chatbot(label="Conversation")
    msg = gr.Textbox(label="Input Message")

    with gr.Row():
        send = gr.Button("➡️ Send")
        clear = gr.Button("🗑️  Clear")

    def user(user_message, history):
        return "", history + [[user_message, None]]

    def bot(history):
        print(history)
        user_message = history[-1][0]
        history[-1][1] = ""
        for character in GEMMA_2_B.stream_complete(user_message):
            history[-1][1] = character.text
            time.sleep(0.05)
            yield history

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(bot, chatbot, chatbot)
    clear.click(lambda: None, None, chatbot, queue=False)
    send.click(user, [msg, chatbot], [msg, chatbot], queue=False).then(bot, chatbot, chatbot)

demo.queue()
demo.launch()