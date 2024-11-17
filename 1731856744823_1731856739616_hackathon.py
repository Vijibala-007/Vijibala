import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from transformers import GPT2LMHeadModel, GPT2Tokenizer
data = pd.read_csv('user_product_ratings.csv')
data.isnull().sum()
data = data.dropna()
data.head()
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
def get_product_embedding(description):
    inputs = tokenizer(description, return_tensors='pt', max_length=512, truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embedding
product_description = data['product_description'][0]
embedding = get_product_embedding(product_description)
interaction_matrix = data.pivot(index='user_id', columns='product_id', values='rating').fillna(0)
cosine_sim = cosine_similarity(interaction_matrix.T)
def recommend_products(product_id, top_n=5):
    product_idx = interaction_matrix.columns.get_loc(product_id)
    sim_scores = list(enumerate(cosine_sim[product_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_products = [interaction_matrix.columns[i] for i, _ in sim_scores[1:top_n+1]]
    return top_products
    recommended_products = recommend_products(123, top_n=5)
print(recommended_products)
def hybrid_recommendation(user_id, top_n=5):
    user_ratings = interaction_matrix.loc[user_id]
    rated_products = user_ratings[user_ratings > 0].index.tolist()
    collaborative_recs = []
    for product_id in rated_products:
      collaborative_recs += recommend_products(product_id, top_n)
      collaborative_recs = list(set(collaborative_recs))
      recommended_embeddings = []
    for product_id in collaborative_recs:
        description = data[data['product_id'] == product_id]['product_description'].values[0]
        recommended_embeddings.append(get_product_embedding(description))
        user_preferences = np.mean([get_product_embedding(data[data['product_id'] == p]['product_description'].values[0]) for p in rated_products], axis=0)
        similarity_scores = [cosine_similarity([user_preferences], [emb])[0][0] for emb in recommended_embeddings]
        top_recommended_idx = np.argsort(similarity_scores)[::-1][:top_n]
        top_recommended_products = [collaborative_recs[i] for i in top_recommended_idx]
    return top_recommended_products
hybrid_recs = hybrid_recommendation(user_id=1, top_n=5)
print(hybrid_recs)
from sklearn.metrics import precision_score, recall_score, f1_score
y_true = [...]
y_pred = hybrid_recommendation(user_id=1, top_n=5)
precision = precision_score(y_true, y_pred, average='binary')
recall = recall_score(y_true, y_pred, average='binary')
f1 = f1_score(y_true, y_pred, average='binary')
print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}")
