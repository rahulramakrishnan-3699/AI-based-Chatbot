import spacy 
import re
import pandas as pd
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

nlp = spacy.load('en_core_web_sm')

df = pd.read_csv('amazon.csv')
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
df = df.dropna(subset=['rating'])
df['rating'] = df['rating'].astype(int)

# Create user-item matrix after fixing duplicates
df = df.groupby(['user_id', 'product_id']).agg({'rating': 'mean'}).reset_index()
user_item_matrix = df.pivot(index='user_id', columns='product_id', values='rating').fillna(0)

# Create sparse matrix
sparse_matrix = csr_matrix(user_item_matrix)

# Train KNN model
knn = NearestNeighbors(metric='cosine', algorithm='brute')
knn.fit(sparse_matrix)

# Load and filter product metadata to match user_item_matrix
product_data = pd.read_csv('amazon.csv')
product_data = product_data[product_data['product_id'].astype(str).isin(user_item_matrix.columns)]

# Create TF-IDF-based product similarity matrix
vectorizer = TfidfVectorizer(stop_words='english')
product_vectors = vectorizer.fit_transform(product_data['category'] + " " + product_data['about_product'])
product_similarity_df = pd.DataFrame(cosine_similarity(product_vectors), 
                                     index=product_data['product_id'], 
                                     columns=product_data['product_id'])

product_categories = ["laptop", "phone", "tablet", "smartwatch", "headphones"]

def extract_keywords(user_input):
    doc = nlp(user_input.lower())
    category = None
    budget = None
    show_ids = False
    for token in doc:
        if token.text in product_categories:
            category = token.text
    
    budget_match = re.search(r'(\d+)[kK]?', user_input)
    if budget_match:
        budget = int(budget_match.group(1))*(1000 if 'k' in user_input else 1)

    if "id" in user_input.lower() or "product code" in user_input.lower():
        show_ids= True

    return category,budget, show_ids

def get_knn_recommendations(user_id):
    user_index = np.random.choice(user_item_matrix.shape[0])  # Selects a row index

    # Fix sparse matrix lookup (use `user_index`, not `user_id`)
    user_vector = sparse_matrix[user_index].toarray().reshape(1, -1)

    # KNN recommendation step
    distances, indices = knn.kneighbors(user_vector, n_neighbors=6)
    recommended_products = user_item_matrix.iloc[indices.flatten()[1:]].mean(axis=0).sort_values(ascending=False).index[:5]

    return recommended_products.tolist()

def get_similar_products(product_id, user_id, top_n=3):
    if product_id in product_similarity_df.index:
        return [p for p in product_similarity_df[product_id].sort_values(ascending=False).index[1:top_n+1] 
                if p in user_item_matrix.columns and user_item_matrix.loc[user_id, p] == 0]
    return []

def get_recommendations(user_id, category = None, budget= None):
    if not category:
        return get_knn_recommendations(user_id)
    category_products = product_data[product_data['category'].str.contains(category, case = False, na= False)]

    if budget:
        category_products = category_products[category_products['discounted_price'] <= budget]

    if category_products.empty:
        return get_knn_recommendations(user_id)
    
    product_id = category_products.sort_values(by="rating", ascending = False)['product_id'].iloc[0]
    return get_similar_products(product_id, user_id,  top_n = 5)

def get_product_details(product_ids, show_ids = False):
    details=[]
    for product_id in product_ids:
        product= product_data[product_data['product_id'] == product_id]
        if not product.empty:
            name = product['product_name'].values[0]
            price = product['discounted_price'].values[0]
            if show_ids:
                details.append(f"{name} - ${price} (ID: {product_id})")
            else:
                details.append(f"{name} - ${price}")
    return details 
    
st.title("AI Chatbot for Product Recomenndations")
st.write("Type your query below to get product suggestions")

if "messages" not in st.session_state:
    st.session_state.messages=[]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

user_input = st.chat_input("Ask me anything...")
if user_input:
    st.session_state.messages.append({"role":"user", "content": user_input})
    user_id = np.random.choice(user_item_matrix.index)
    category, budget, show_ids = extract_keywords(user_input)
    recommendations = get_recommendations (user_id, category, budget)

    if not recommendations:
        chatbot_response = "I couldnt find any recommendations. Try asking for another query!"
    else:
        product_details = get_product_details(recommendations, show_ids)
        if product_details:
            recommendations_text = "\n".join([f"-{product}" for product in product_details])
            chatbot_response = f"Here are some {category if category else 'top'} recommendations{f'under ${budget}' if budget else ''}: {', '.join(product_details)}"

    st.session_state.messages.append({"role":"user", "content":user_input})
    st.session_state.messages.append({"role":"assistant", "content": chatbot_response})

    with st.chat_message("assistant"):
        st.write(chatbot_response)