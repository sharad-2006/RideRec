from flask import Flask, render_template, request
import pandas as pd
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

app = Flask(__name__)


def clean_text(text):
    if type(text) != str:
        return ''
    text = text.lower()
    text = re.sub(r'<.*?>', '', text) 
    text = re.sub(r'[^a-zA-Z\s]', '', text)  
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    words_no_stop = [word for word in words if word not in stop_words]
    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(word) for word in words_no_stop]
    return ' '.join(stemmed_words)


def make_vectors(descriptions, user_search):
    search_words = clean_text(user_search).split()
    vectorizer = TfidfVectorizer(vocabulary=search_words)
    tfidf_matrix = vectorizer.fit_transform(descriptions)
    return tfidf_matrix, vectorizer


def get_similarity(user_vector, bike_vectors):
    similarities = cosine_similarity(user_vector, bike_vectors)
    return similarities.flatten()


def get_simple_score(row, user_engine_cc, user_price):
    cc_diff = 0 if user_engine_cc is None else abs(row['engine_cc'] - user_engine_cc) / 1000
    price_diff = 0 if user_price is None else abs(row['price'] - user_price) / 100000
    return 0.6 * cc_diff + 0.4 * price_diff


def rank_bikes_simple(bikes_df, similarities, user_engine_cc, user_price):
    bikes_df1 = bikes_df.copy()
    bikes_df1['similarity_score'] = similarities
    bikes_df1['rule_score'] = bikes_df1.apply(
        lambda row: get_simple_score(row, user_engine_cc, user_price),
        axis=1
    )
    bikes_df1['final_score'] = 0.8 * bikes_df1['similarity_score'] - 0.2 * bikes_df1['rule_score']
    return bikes_df1.sort_values('final_score', ascending=False)


def load_bike_data(csv_path):
    if not os.path.exists(csv_path):
        print(f"Error: File '{csv_path}' not found.")
        exit(1)
    df = pd.read_csv(csv_path)
    required_columns = ['bike_id', 'brand', 'model', 'engine_cc', 'price', 'description']
    for col in required_columns:
        if col not in df.columns:
            print(f"Missing column: {col}")
            exit(1)
    df['clean_desc'] = df['description'].apply(clean_text)
    return df

@app.route('/', methods=['GET', 'POST'])
def index():
    top_bikes_display = None
    query = ''
    engine_cc_input = ''
    price_input = ''
    form_submitted = False

    if request.method == 'POST':
        form_submitted = True
       
        query = request.form.get('query', '').strip()
        engine_cc_input = request.form.get('engine_cc', '').strip().lower()
        price_input = request.form.get('price', '').strip().lower()

        user_engine_cc = int(engine_cc_input) if engine_cc_input and engine_cc_input != 'cc' else None
        user_price = int(price_input) if price_input and price_input != 'price' else None

        bikes_df = load_bike_data('recbike.csv')
        bike_vectors, vectorizer = make_vectors(bikes_df['clean_desc'], query)
        user_vector = vectorizer.transform([clean_text(query)])

        similarities = get_similarity(user_vector, bike_vectors)
        ranked_bikes = rank_bikes_simple(bikes_df, similarities, user_engine_cc, user_price)

        top_bikes = ranked_bikes.head(5).copy()
        top_bikes['price'] = top_bikes['price'].apply(lambda x: f"₹{x:,.0f}")
        top_bikes['final_score'] = top_bikes['final_score'].apply(lambda x: f"{x:.4f}")

        top_bikes_display = top_bikes

    
    return render_template(
        'index.html',
        top_bikes=top_bikes_display,
        query=query,
        engine_cc_input=engine_cc_input,
        price_input=price_input,
        form_submitted=form_submitted
    )

if __name__ == '__main__':
    app.run(debug=True)
