import os
from dotenv import load_dotenv

from ModelCalls import GPT4Analysis
C_FOLDER_PATH = "./llms-for-code-analysis-main/dataset/nonobfuscated/c-data"
JAVASCRIPT_FOLDER_PATH = "./llms-for-code-analysis-main/dataset/nonobfuscated/javascript-data"
PYTHON_FOLDER_PATH = "./llms-for-code-analysis-main/dataset/nonobfuscated/python-data"

prompts = []
filenames = []
load_dotenv()
# Getting C Metrics
for filename in os.listdir(C_FOLDER_PATH):
    file_path = os.path.join(C_FOLDER_PATH, filename)
    if os.path.isdir(file_path):
        continue
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        print(f"Skipping {filename} (error reading file: {e})")
        continue

    prompt = f"Analyze and tell me what this file does. Code: {filename}\n\n{content}"
    prompts.append(prompt)
    filenames.append(filename)

CAnalysis = GPT4Analysis(prompts, filenames, "C")


prompts = []
filenames = []
# Getting JavaScript Metrics
for filename in os.listdir(JAVASCRIPT_FOLDER_PATH):
    file_path = os.path.join(JAVASCRIPT_FOLDER_PATH, filename)
    if os.path.isdir(file_path):
        continue
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        print(f"Skipping {filename} (error reading file: {e})")
        continue

    prompt = f"Analyze and tell me what this file does. Code: {filename}\n\n{content}"
    prompts.append(prompt)
    filenames.append(filename)

JAnalysis = GPT4Analysis(prompts, filenames, "JavaScript")



prompts = []
filenames = []
# Getting Python Metrics
for filename in os.listdir(PYTHON_FOLDER_PATH):
    file_path = os.path.join(PYTHON_FOLDER_PATH, filename)
    if os.path.isdir(file_path):
        continue
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        print(f"Skipping {filename} (error reading file: {e})")
        continue

    prompt = f"Analyze and tell me what this file does. Code: {filename}\n\n{content}"
    prompts.append(prompt)
    filenames.append(filename)

PAnalysis = GPT4Analysis(prompts, filenames, "Python")