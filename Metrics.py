from urllib import response
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ModelCalls import GPT4Similarity


sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

# COSINE SIMILARITY BETWEEN TWO TEXTS USING TF-IDF VECTORIZER
def CosineSimilarity(text1, text2):
    vec = TfidfVectorizer().fit_transform([text1, text2])
    return float(cosine_similarity(vec[0], vec[1])[0][0])

# SEMANTIC SIMILARITY BETWEEN TWO TEXTS USING PRETRAINED BERT MODEL
def WebBertSim(text1, text2):
    emb1 = sbert_model.encode(text1, convert_to_tensor=True)
    emb2 = sbert_model.encode(text2, convert_to_tensor=True)
    cosine_score = float(util.pytorch_cos_sim(emb1, emb2))
    return 1 + ((cosine_score + 1) * 4 / 2)  # scale [-1,1] → [1,5]

# SIMILARITY BETWEEN TWO TEXTS USING GPT-4
def GPT4AllSim(text1, text2):
    response = GPT4Similarity(text1, text2)
    if response.startswith("Yes"):
        return 1.0
    else:
        return 0.0

# SUCCESS RATE CALCULATION
def SuccessRate(successes, total):
    if total == 0:
        return 0.0
    return float(successes) / float(total)



