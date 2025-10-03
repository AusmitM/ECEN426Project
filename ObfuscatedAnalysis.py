from semantic_text_similarity.models import WebBertSimilarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# texts = [text1, text2]
# vec = TfidfVectorizer().fit_transform(texts)  # sparse matrix
# sim_matrix = cosine_similarity(vec[0], vec[1])
# print("TF-IDF cosine:", float(sim_matrix))


# This file uses the Obfuscated code from the 100 contest winner projects after the 2011 IOCCC competition and instruct the LLM to preform deobfuscation.
# The original obfuscated code is given to the LLM without any modification

import re
from openai import OpenAI
from dotenv import load_dotenv
import subprocess
import tempfile
import os
import os

from Metrics import CosineSimilarity, WebBertSim, GPT4AllSim
from ModelCalls import GPT35TurboAnalysis

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))



def GPT41Analysis(filepath):
    response = client.chat.completions.create(
    model="gpt-4.1",
    messages=[
        {
            "role": "user",
            "content": f"Analyze the code and tell me what it does. Code: {filepath}"
        }
    ]
    )
    return response.choices[0].message.content

# We will assume that the parameter allObfuscationFiles is structured like the following
# [ Array of Regularly Obfuscated Files,
#  Array of Control Flow Flattened Files, 
#  Array of Dead Code Injected Files,
#  Array of String Splitted Files,
#  Array of Wobfuscated Files]
def metricsCalculator(allObfuscatedFiles):
    cosineSimArray = []
    bertSimArray = []
    GPTSimArray = []
    for obfuscationMethod in allObfuscatedFiles:
        gpt4CosSum = 0
        gpt35CosSum = 0
        gpt4BertSum = 0
        gpt35BertSum = 0
        gpt4GPTSum = 0
        gpt35GPTSum = 0
        for currentFilePath in obfuscationMethod:
            gpt4analysis = GPT41Analysis(currentFilePath)
            gpt35analysis = GPT35TurboAnalysis(currentFilePath)
            gpt4CosSum += CosineSimilarity(answerText, gpt4analysis)
            gpt35CosSum += CosineSimilarity(answerText, gpt35analysis)
            gpt4BertSum += WebBertSim(answerText, gpt4analysis)
            gpt35BertSum += WebBertSim(answerText, gpt35analysis)
            gpt4GPTSum += GPT4AllSim(answerText, gpt4analysis)
            gpt35GPTSum += GPT4AllSim(answerText, gpt35analysis)
        cosineSimArray.append([gpt35CosSum/len(obfuscationMethod), gpt4CosSum/len(obfuscationMethod)])
        bertSimArray.append([gpt35BertSum/len(obfuscationMethod), gpt4BertSum/len(obfuscationMethod)])
        GPTSimArray.append([gpt35GPTSum/len(obfuscationMethod), gpt4GPTSum/len(obfuscationMethod)])
    gpt35OverallCosSum = 0
    gpt4OverallCosSum = 0
    gpt35OverallBertSum = 0
    gpt4OverallBertSum = 0
    gpt35OverallGPTSum = 0
    gpt4OverallGPTSum = 0
    for i in range(5):
        gpt35OverallCosSum += cosineSimArray[0][0]
        gpt4OverallCosSum += cosineSimArray[0][1]
        gpt35OverallBertSum += bertSimArray[0][0]
        gpt4OverallBertSum += bertSimArray[0][1]
        gpt35OverallGPTSum += GPTSimArray[0][0]
        gpt4OverallGPTSum += GPTSimArray[0][1]
    bertSimArray.append([gpt35OverallBertSum/5, gpt4OverallBertSum/5])
    cosineSimArray.append([gpt35OverallCosSum/5, gpt4OverallCosSum/5])
    GPTSimArray.append([gpt35OverallGPTSum/5, gpt4OverallGPTSum/5])
    
    return (cosineSimArray, bertSimArray, GPTSimArray)



def GPT35TurboDeobfuscation(filepath):
    response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {
            "role": "user",
            "content": f"““You are an expert in code analysis. De-obfuscate the code and generate a readable new version. Code: {filepath}"
        }
    ]
    )
    return response.choices[0].message.content

def GPT41Deobfuscation(filepath):
    response = client.chat.completions.create(
    model="gpt-4.1",
    messages=[
        {
            "role": "user",
            "content": f"““You are an expert in code analysis. De-obfuscate the code and generate a readable new version. Code: {filepath}"
        }
    ]
    )
    return response.choices[0].message.content

# The parameter filepaths should contain all the IOCCC files

def deobfuscationMetrics(filepaths):
    gpt4generated = 0
    gpt35generated = 0
    gpt4Compile = 0
    gpt35Compile = 0
    for currentFile in filepaths:
        gpt35code = GPT35TurboDeobfuscation(currentFile)
        gpt4code = GPT41Deobfuscation(currentFile)
        if ("```" in gpt35code):
            gpt35generated += 1
            match = re.search(r'```(.*?)```', gpt35code, re.DOTALL)
            if match:
                code = match.group(1).strip()
                if (checkCompile(code)):
                    gpt35Compile += 1
        if ("```" in gpt4code):
            gpt4generated += 1
            match = re.search(r'```(.*?)```', gpt4code, re.DOTALL)
            if match:
                code = match.group(1).strip()
                if (checkCompile(code)):
                    gpt4Compile += 1
    gpt35generation = gpt35generated / len(filepaths)
    gpt35Compilation = gpt35Compile / len(filepaths)
    gpt4generation = gpt4generated / len(filepaths)
    gpt4Compilation = gpt4Compile / len(filepaths)

    return (gpt35generation, gpt4generation, gpt35Compilation, gpt4Compilation)

# This checkCompile function only checks if c code is compilable. 
# For other languages, we can easily extend this functionality by first check if the string Python/C/cpp exists in the API response and feeding the code to the respective compiler function. 
def checkCompile(code):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.c', delete=False) as f:
        f.write(code)
        c_file_path = f.name
    
    try:
        result = subprocess.run(
            ['gcc', c_file_path, '-o', "program"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        # Check if compilation was successful
        return (result.returncode == 0)
    finally:
        # Clean up the temporary C file
        if os.path.exists(c_file_path):
            os.remove(c_file_path)