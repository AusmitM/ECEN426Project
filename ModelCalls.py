from openai import OpenAI
from dotenv import load_dotenv
import os

import torch
from transformers import pipeline

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# SIMILARITY BETWEEN TWO TEXTS USING GPT-4
def GPT4Similarity(text1, text2):
    response = client.chat.completions.create(
    model="gpt-4.1",
    messages=[
        {
            "role": "user",
            "content": f"You are given two descriptions of two code snippets: Description1: {text1} and Description2: {text2}. Corresponding code snippets are not available. From the given text, do you think the two descriptions correspond to two code snippets with roughly similar functionalities? Output should be \"Yes\" if similar, or \"No\" otherwise, followed by a brief justification of how this is determined."
        }
    ]
    )
    return response.choices[0].message.content

# ANALYSIS OF A CODE FILE USING STARCHAT MODEL
def StarChatAnalysis():
    pipe = pipeline("text-generation", model="HuggingFaceH4/starchat-beta", torch_dtype=torch.bfloat16, device_map="auto")
    prompt_template = "<|system|>\n<|end|>\n<|user|>\n{query}<|end|>\n<|assistant|>"
    prompt = prompt_template.format(query="How do I sort a list in Python?")
    # We use a special <|end|> token with ID 49155 to denote ends of a turn
    outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.2, top_k=50, top_p=0.95, eos_token_id=49155)
    print(outputs[0]['generated_text'])
    

# ANALYSIS OF A CODE FILE USING GPT-3.5-TURBO
def GPT35TurboAnalysis(filepath):
    response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {
            "role": "user",
            "content": f"Analyze the code and tell me what it does. Code: {filepath}"
        }
    ]
    )
    return response.choices[0].message.content
# For manual checking, we first make a code analysis API call to GPT-4. 
# We will manually check the explanation generated and if it is correct, we will mark that explanation as the GRM for evaluating the correctness of explanations generated from other models.
# The similarity between the explanations are generated using similarity models like GPT4Similarity, Bert similarity score, and cosine similarity score


