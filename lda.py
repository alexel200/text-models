import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim import corpora, models, similarities
import time

CATEGORY = "Sports"

# Ensure the necessary NLTK data is downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Load the data
file_path = 'dataset/news1.csv'
data = pd.read_csv(file_path)

# Extract the relevant columns
articles = data['description'].fillna("").astype(str).tolist()
topics = data['article_section'].tolist()

# Step 1: Preprocessing and Tokenization
stop_words = set(stopwords.words('english'))
texts = [[word for word in word_tokenize(article.lower()) if word.isalnum() and word not in stop_words]
         for article in articles]

# Step 2: Calculate the LDA Vectors
start_time = time.time()
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
lda_model = models.LdaModel(corpus, num_topics=30, passes=2, random_state=42)
lda_vectors = [lda_model[doc] for doc in corpus]
index = similarities.MatrixSimilarity(lda_vectors)
model_creation_time = time.time() - start_time

# Step 3: Calculate the Ratio Quality
total_goods = 0
num_articles_food_and_drink = sum([1 for t in topics if t == CATEGORY])

for i, article in enumerate(articles):
    if topics[i] == CATEGORY:
        start_time = time.time()
        sims = index[lda_vectors[i]]
        top_10_indices = sims.argsort()[-11:-1][::-1]  # Exclude the article itself
        goods = sum([1 for index in top_10_indices if topics[index] == CATEGORY])
        total_goods += goods
        pseudocode_execution_time = time.time() - start_time

ratio_quality = total_goods / (num_articles_food_and_drink * 10)

print(f"Ratio Quality: {ratio_quality}")
print(f"Model Creation Time: {model_creation_time} seconds")
print(f"Pseudocode Execution Time: {pseudocode_execution_time} seconds per article")