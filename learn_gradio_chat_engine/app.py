"""
This gradio demo is used to test the streaming text generation of the model


As the it will produce the output in real time, the model will be able to generate the text in real time.

We will be using gemma-2 model for this.
"""

import time
import os
import gradio as gr

from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core import Settings
# setup prompts - specific to StableLM
from llama_index.core import PromptTemplate

# SEE: https://huggingface.co/docs/hub/security-tokens
# We just need a token with read permissions for this demo
HF_TOKEN = os.getenv("HF_TOKEN")

system_prompt = ""
query_wrapper_prompt = PromptTemplate("""<start_of_turn>user
                                      {query_str}<end_of_turn>
                                      <start_of_turn>model
                                      """)


llm = HuggingFaceLLM(tokenizer_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                      model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                      query_wrapper_prompt=query_wrapper_prompt,
                      device_map="cuda",
                      stopping_ids=[1, 0, 20225],
                      generate_kwargs={"temperature": 0.7, "do_sample": False, "top_k": 50, "top_p": 0.95},
                      max_new_tokens=256, 
                    )
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
        for character in llm.stream_complete(user_message):
            history[-1][1] = character.text
            time.sleep(0.05)
            yield history

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(bot, chatbot, chatbot)
    clear.click(lambda: None, None, chatbot, queue=False)
    send.click(user, [msg, chatbot], [msg, chatbot], queue=False).then(bot, chatbot, chatbot)

demo.queue()
demo.launch()