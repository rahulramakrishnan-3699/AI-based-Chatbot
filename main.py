import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load and preprocess data
df = pd.read_csv('amazon.csv')
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
df = df.dropna(subset=['rating'])
df['rating'] = df['rating'].astype(int)

# Create user-item matrix after fixing duplicates
df = df.groupby(['user_id', 'product_id']).agg({'rating': 'mean'}).reset_index()
user_item_matrix = df.pivot(index='user_id', columns='product_id', values='rating').fillna(0)
print("User-Item Matrix Shape:", user_item_matrix.shape)

# Create sparse matrix
sparse_matrix = csr_matrix(user_item_matrix)

# Train KNN model
knn = NearestNeighbors(metric='cosine', algorithm='brute')
knn.fit(sparse_matrix)

# Get recommendations for a random user
user_index = np.random.choice(user_item_matrix.shape[0])  # Selects a row index
user_id = user_item_matrix.index[user_index]  # Get actual user ID

# Fix sparse matrix lookup (use `user_index`, not `user_id`)
user_vector = sparse_matrix[user_index].toarray().reshape(1, -1)

# KNN recommendation step
distances, indices = knn.kneighbors(user_vector, n_neighbors=6)
recommended_products = user_item_matrix.iloc[indices.flatten()[1:]].mean(axis=0).sort_values(ascending=False).index[:5]

print("Recommended Products:", recommended_products.tolist())

# Load and filter product metadata to match user_item_matrix
product_data = pd.read_csv('amazon.csv')
product_data = product_data[product_data['product_id'].astype(str).isin(user_item_matrix.columns)]

# Create TF-IDF-based product similarity matrix
vectorizer = TfidfVectorizer(stop_words='english')
product_vectors = vectorizer.fit_transform(product_data['category'] + " " + product_data['about_product'])
product_similarity_df = pd.DataFrame(cosine_similarity(product_vectors), 
                                     index=product_data['product_id'], 
                                     columns=product_data['product_id'])

# Function to get similar products (ensures only unrated ones are recommended)
def get_similar_products(product_id, top_n=3):
    if product_id in product_similarity_df.index:
        return [p for p in product_similarity_df[product_id].sort_values(ascending=False).index[1:top_n+1] 
                if p in user_item_matrix.columns and user_item_matrix.loc[user_id, p] == 0]
    return []

# Ensure recommended products are valid and unrated by the user
filtered_recommended_products = [p for p in recommended_products 
                                 if p in user_item_matrix.columns and user_item_matrix.loc[user_id, p] == 0]

# Generate final recommendations
final_recommendations = []
for product_id in filtered_recommended_products:
    similar_products = get_similar_products(product_id)
    final_recommendations.extend(similar_products if similar_products else [product_id])

# Ensure at least 5 recommendations by filling with top-rated unrated products
if len(final_recommendations) < 5:
    backup_recommendations = [p for p in user_item_matrix.mean().sort_values(ascending=False).index 
                              if user_item_matrix.loc[user_id, p] == 0 and p not in final_recommendations]
    final_recommendations.extend(backup_recommendations[:5 - len(final_recommendations)])

# Return unique top 5 recommendations
final_recommendations = list(set(final_recommendations))[:5]

print("Final Recommended Products (KNN + Content-Based):", final_recommendations)
print("✅ Hybrid KNN Model is ready!")
