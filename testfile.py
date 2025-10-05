import os
from xmlrpc import client

import torch
from Metrics import CosineSimilarity, GPT4AllSim, WebBertSim
from ModelCalls import StarChatAnalysis


# text1 = "I love dogs"
# text2 = "I really really really like dogs"

# print("TF-IDF cosine:", CosineSimilarity(text1, text2))
# print("WebBert similarity:", WebBertSim(text1, text2))

# test1 = "This function takes in a list of numbers and returns the sum of all the even numbers in tghe list."
# test2 = "This function takes in a list of numbers and returns the sum of all the odd numbers in the list."
# print("GPT-4 similarity:", GPT4AllSim(test1, test2))

# StarChatAnalysis()





# Use a pipeline as a high-level helper
# Use a pipeline as a high-level helper
from huggingface_hub import InferenceClient

# Initialize the inference client
from huggingface_hub import InferenceClient

# Initialize the inference client
from huggingface_hub import InferenceClient

# client = InferenceClient("codellama/CodeLlama-13b-Instruct-hf")

# response = client.text_generation(
#     prompt="You are a helpful scientific assistant. Explain the difference between TCP and UDP.",
#     max_new_tokens=500,
# )

# print(response)



# from transformers import pipeline

# from transformers import pipeline

# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch

# model_name = "HuggingFaceH4/starchat-beta"
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     device_map="auto",   # automatically split layers across GPUs
#     load_in_4bit=True    # quantized 4-bit
# )

# from transformers import pipeline

# generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# result = generator("Explain TCP vs UDP", max_new_tokens=200)
# print(result[0]['generated_text'])



from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Model name
model_name = "meta-llama/CodeLlama-13b-Instruct-hf"

print("🔹 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("🔹 Loading 4-bit quantized model (this may take a minute)...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",          # Automatically assign layers across GPU/CPU
    load_in_4bit=True,          # Quantize to 4-bit
    torch_dtype=torch.float16,  # Mixed precision
    low_cpu_mem_usage=True,     # Reduces RAM pressure
)

print("✅ Model loaded successfully!")

# Create a text generation pipeline
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device_map="auto",
)

# Example prompt
prompt = """\
You are an expert Python programmer.
Write a function that takes a list of integers and returns a new list \
containing only the even numbers, sorted in descending order.
"""

print("🧠 Generating text...")
outputs = generator(
    prompt,
    max_new_tokens=200,
    temperature=0.7,
    do_sample=True,
    top_p=0.9,
    repetition_penalty=1.1,
)

print("\n📝 Generated Output:\n")
print(outputs[0]["generated_text"])

