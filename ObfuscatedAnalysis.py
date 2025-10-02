from semantic_text_similarity.models import WebBertSimilarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# texts = [text1, text2]
# vec = TfidfVectorizer().fit_transform(texts)  # sparse matrix
# sim_matrix = cosine_similarity(vec[0], vec[1])
# print("TF-IDF cosine:", float(sim_matrix))
