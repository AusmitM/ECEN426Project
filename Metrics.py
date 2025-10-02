from urllib import response
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from dotenv import load_dotenv
import os
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

def CosineSimilarity(text1, text2):
    vec = TfidfVectorizer().fit_transform([text1, text2])
    return float(cosine_similarity(vec[0], vec[1])[0][0])

def WebBertSim(text1, text2):
    emb1 = sbert_model.encode(text1, convert_to_tensor=True)
    emb2 = sbert_model.encode(text2, convert_to_tensor=True)
    cosine_score = float(util.pytorch_cos_sim(emb1, emb2))
    return 1 + ((cosine_score + 1) * 4 / 2)  # scale [-1,1] → [1,5]

def GPT4AllSim(text1, text2):
    response = client.chat.completions.create(
    model="gpt-4.1",
    messages=[
        {
            "role": "user",
            "content": f"You are given two descriptions of two code snippets: Description1: {text1} and Description2: {text2}. Corresponding code snippets are not available. From the given text, do you think the two descriptions correspond to two code snippets with roughly similar functionalities? Output should be \"Yes\" if similar, or \"No\" otherwise, followed by a brief justification of how this is determined."
        }
    ]
    )
    if response.choices[0].message.content.startswith("Yes"):
        return 1.0
    else:
        return 0.0
    # print(response.choices[0].message.content)

def SuccessRate(successes, total):
    if total == 0:
        return 0.0
    return float(successes) / float(total)



