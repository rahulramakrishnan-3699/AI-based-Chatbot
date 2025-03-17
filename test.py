import spacy 
import re

nlp = spacy.load('en_core_web_sm')

product_categories = ["laptop", "phone", "tablet", "smartwatch", "headphone"]

def extract_keywords(user_input):
    doc = nlp(user_input.lower())
    category = None
    budget = None
    for token in doc:
        if token.text in product_categories:
            categoru = token.text
    
    budget_match = re.search(r'(\d+)[kK]?', user_input)
    if budget_match:
        budget = int(budget_match.group(1))*(1000 if 'k' in user_input else 1)

    return category,budget

user_queries = [
    "Can you recommend a laptip below $1000",
    "I need a phone below $500.",
    "Suggest some headphones under 1k.",
    "What are the best smartwatches?"
]

for query in user_queries:
    category, budget = extract_keywords(query)
    print(f"User Query: {query}")
    print(f" Extracted Category: {category}, Budget: {budget}")
    print("-" * 50)