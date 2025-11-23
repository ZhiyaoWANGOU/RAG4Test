# ---------------------------------------------------------
# Text Similarity Comparison: TF-IDF vs BERT vs SBERT
# ---------------------------------------------------------

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, BertModel
import torch


# ---------------------------------------------------------
# Example sentences (traceability scenario)
# ---------------------------------------------------------
sent1 = "The application fails to load when the user has a weak network connection."
sent2 = "The application fails to login when the user has invalid credentials."

sentences = [sent1, sent2]


# ---------------------------------------------------------
# 1. TF-IDF Similarity
# ---------------------------------------------------------
def tfidf_similarity(s1, s2):
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([s1, s2])
    sim = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    return float(sim)


# ---------------------------------------------------------
# 2. BERT (CLS embedding)
# ---------------------------------------------------------
def bert_cls_embedding(texts):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")

    encoded = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**encoded)

    # CLS token = outputs.last_hidden_state[:, 0, :]
    return outputs.last_hidden_state[:, 0, :].numpy()


def cosine(u, v):
    return float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))


# ---------------------------------------------------------
# 3. SBERT
# ---------------------------------------------------------
def sbert_embedding(texts):
    model = SentenceTransformer("all-mpnet-base-v2")
    return model.encode(texts)


# ---------------------------------------------------------
# Run All Methods
# ---------------------------------------------------------
print("\n==================== RESULTS ====================")

# 1. TF-IDF
sim_tfidf = tfidf_similarity(sent1, sent2)
print(f"TF-IDF similarity:       {sim_tfidf:.4f}")

# 2. BERT
bert_vecs = bert_cls_embedding(sentences)
sim_bert = cosine(bert_vecs[0], bert_vecs[1])
print(f"BERT (CLS) similarity:   {sim_bert:.4f}")

# 3. SBERT
sbert_vecs = sbert_embedding(sentences)
sim_sbert = cosine(sbert_vecs[0], sbert_vecs[1])
print(f"SBERT similarity:        {sim_sbert:.4f}")

print("=================================================\n")