import os
from dotenv import load_dotenv

from ModelCalls import CodeLlama2Analysis, GPT35TurboAnalysis, Llama2Analysis, StarChatAnalysis
C_FOLDER_PATH = "./llms-for-code-analysis-main/dataset/nonobfuscated/c-data"
JAVASCRIPT_FOLDER_PATH = "./llms-for-code-analysis-main/dataset/nonobfuscated/javascript-data"
PYTHON_FOLDER_PATH = "./llms-for-code-analysis-main/dataset/nonobfuscated/python-data"

prompts = []
# C Metrics
CStarChatMetrics = []
CGPT35TurboMetrics = []
CLlama2Metrics = []
CCodeLlama2Metrics = []
# JavaScript Metrics
JStarChatMetrics = []
JGPT35TurboMetrics = []
JLlama2Metrics = []
JCodeLlama2Metrics = []
# Python Metrics
PStarChatMetrics = []
PGPT35TurboMetrics = []
PLlama2Metrics = []
PCodeLlama2Metrics = []
# Overall Metrics
OverallStarChatMetrics = []
OverallGPT35TurboMetrics = []
OverallLlama2Metrics = []
OverallCodeLlama2Metrics = []


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

    CStarChatMetrics = StarChatAnalysis(prompts)
    CGPT35TurboMetrics = GPT35TurboAnalysis(prompts)
    CLlama2Metrics = Llama2Analysis(prompts)
    CCodeLlama2Metrics = CodeLlama2Analysis(prompts)


# Getting JavaScript Metrics
prompts = []
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

    JStarChatMetrics = StarChatAnalysis(prompts)
    JGPT35TurboMetrics = GPT35TurboAnalysis(prompts)
    JLlama2Metrics = Llama2Analysis(prompts)
    JCodeLlama2Metrics = CodeLlama2Analysis(prompts)


# Getting Python Metrics
prompts = []
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

    PStarChatMetrics = StarChatAnalysis(prompts)
    PGPT35TurboMetrics = GPT35TurboAnalysis(prompts)
    PLlama2Metrics = Llama2Analysis(prompts)
    PCodeLlama2Metrics = CodeLlama2Analysis(prompts)


# Assuming the outputs of each metrics list is in the order of [Cosine, Bert, GPT4All]
# Each list should be in the form of [GPT3.5, Llama-2B, Code-Llama2, StarChat] metrics, of which should be in [C, JavaScript, Python, Overall] order
CosineMetrics = [[CGPT35TurboMetrics[0], JGPT35TurboMetrics[0], PGPT35TurboMetrics[0], (CGPT35TurboMetrics[0] + JGPT35TurboMetrics[0] + PGPT35TurboMetrics[0]) / 3],
                 [CLlama2Metrics[0], JLlama2Metrics[0], PLlama2Metrics[0], (CLlama2Metrics[0] + JLlama2Metrics[0] + PLlama2Metrics[0]) / 3],
                 [CCodeLlama2Metrics[0], JCodeLlama2Metrics[0], PCodeLlama2Metrics[0], (CCodeLlama2Metrics[0] + JCodeLlama2Metrics[0] + PCodeLlama2Metrics[0]) / 3],
                 [CStarChatMetrics[0], JStarChatMetrics[0], PStarChatMetrics[0], (CStarChatMetrics[0] + JStarChatMetrics[0] + PStarChatMetrics[0]) / 3]]
BertMetrics = [[CGPT35TurboMetrics[1], JGPT35TurboMetrics[1], PGPT35TurboMetrics[1], (CGPT35TurboMetrics[1] + JGPT35TurboMetrics[1] + PGPT35TurboMetrics[1]) / 3],
               [CLlama2Metrics[1], JLlama2Metrics[1], PLlama2Metrics[1], (CLlama2Metrics[1] + JLlama2Metrics[1] + PLlama2Metrics[1]) / 3],
               [CCodeLlama2Metrics[1], JCodeLlama2Metrics[1], PCodeLlama2Metrics[1], (CCodeLlama2Metrics[1] + JCodeLlama2Metrics[1] + PCodeLlama2Metrics[1]) / 3],
               [CStarChatMetrics[1], JStarChatMetrics[1], PStarChatMetrics[1], (CStarChatMetrics[1] + JStarChatMetrics[1] + PStarChatMetrics[1]) / 3]]
GPT4AllMetrics = [[CGPT35TurboMetrics[2], JGPT35TurboMetrics[2], PGPT35TurboMetrics[2], (CGPT35TurboMetrics[2] + JGPT35TurboMetrics[2] + PGPT35TurboMetrics[2]) / 3],
                  [CLlama2Metrics[2], JLlama2Metrics[2], PLlama2Metrics[2], (CLlama2Metrics[2] + JLlama2Metrics[2] + PLlama2Metrics[2]) / 3],
                  [CCodeLlama2Metrics[2], JCodeLlama2Metrics[2], PCodeLlama2Metrics[2], (CCodeLlama2Metrics[2] + JCodeLlama2Metrics[2] + PCodeLlama2Metrics[2]) / 3],
                  [CStarChatMetrics[2], JStarChatMetrics[2], PStarChatMetrics[2], (CStarChatMetrics[2] + JStarChatMetrics[2] + PStarChatMetrics[2]) / 3]]

