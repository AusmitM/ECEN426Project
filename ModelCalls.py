from openai import OpenAI
from dotenv import load_dotenv
import os

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from transformers import pipeline
from huggingface_hub import InferenceClient

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

# SIMILARITY BETWEEN TWO TEXTS USING GPT-4
def GPT4Analysis(prompts, filenames, language):
    output_folder = "./GPT4OUTPUTS"
    if language == "C":
        output_folder = "./CGPT4OUTPUTS"
    elif language == "JavaScript":
        output_folder = "./JavaScriptGPT4OUTPUTS"
    elif language == "Python":
        output_folder = "./PythonGPT4OUTPUTS"

        
    all_messages = []
    for prompt in prompts:
        all_messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
    model="gpt-4.1",
    messages=all_messages
    )

    # Save each response to a separate file
    for filename in enumerate(filenames):
        output_path = os.path.join(output_folder, f"{filename}_GPT4_analysis.txt")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(response.choices[0].message.content)

# ANALYSIS OF A CODE FILE USING STARCHAT MODEL
def StarChatAnalysis(prompts):
    model_name = "HuggingFaceH4/starchat-beta"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",   # automatically split layers across GPUs
        load_in_4bit=True    # quantized 4-bit
    )
    
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    result = generator("Explain TCP vs UDP", max_new_tokens=200)
    print(result[0]['generated_text'])
    

# ANALYSIS OF A CODE FILE USING GPT-3.5-TURBO
def GPT35TurboAnalysis(prompts):
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


# ANALYSIS OF A CODE FILE USING LLAMA2
def Llama2Analysis(prompts):
    client = InferenceClient("meta-llama/Llama-2-13b-chat-hf")
    response = client.chat_completion(
        model="meta-llama/Llama-2-13b-chat-hf",
        messages=[
            {"role": "system", "content": "You are a helpful scientific assistant."}
        ],
        max_tokens=200
    )

    print(response.choices[0].message["content"])
# ANALYSIS OF A CODE FILE USING CODELLAMA2
def CodeLlama2Analysis(prompts):
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


# For manual checking, we first make a code analysis API call to GPT-4. 
# We will manually check the explanation generated and if it is correct, we will mark that explanation as the GRM for evaluating the correctness of explanations generated from other models.
# The similarity between the explanations are generated using similarity models like GPT4Similarity, Bert similarity score, and cosine similarity score


