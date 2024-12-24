from flask import Flask, render_template, request, jsonify
import os
import io
import base64
import pickle
import nltk
import string
import pandas as pd
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

stop_words = set(stopwords.words('english'))
# tfidf = TfidfVectorizer(max_features=5000)
app = Flask(__name__)

nltk.download('stopwords')
nltk.download('punk_tab')

def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    tokens = [word for word in word_tokenize(text) if word not in stop_words]
    return ' '.join(tokens)

def generate_wordcloud(text):
    # Generate the word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

    # Save the word cloud to a BytesIO object
    img = io.BytesIO()
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.savefig(img, format='png', bbox_inches='tight')
    plt.close()
    img.seek(0)

    # Encode the image in base64 to send to the frontend
    return base64.b64encode(img.getvalue()).decode('utf-8')

@app.route('/')
def index():        
    return render_template('main.html')

@app.route('/result', methods=['POST'])
def predict():
    label = {
        0: 'Anxiety',
        1: 'Bipolar',
        2: 'Depression',
        3: 'Normal',
        4: 'Personality Disorder',
        5: 'Stress',
        6: 'Suicidal'
    }

    # Load the saved model and vectorizer pipeline
    with open('model_and_vectorizer_75%.pkl', 'rb') as f:
        loaded_pipeline = pickle.load(f)

    # Initialize variables
    user_input = None
    prediction_counts = None
    type = None
    data = 100  
    Positive = 0
    Negative = 0
    results = [] 

    if 'single' in request.form:
        type = 'text'
        user_input = request.form['text']
        cleaned_input = clean_text(user_input)

        prediction = loaded_pipeline.predict([cleaned_input])[0]
        predicted_label = label[prediction]
        
        results.append((user_input, predicted_label))

        Positive = 100 if predicted_label == 'Normal' else 0
        Negative = 0 if predicted_label == 'Normal' else 100
        
        wordcloud_image = generate_wordcloud(user_input)

    elif 'csv_file' in request.form:

        type = 'file'
        file = request.files['csv']

        df = pd.read_csv(file)
        print(df.columns) 


        df['cleaned_statement'] = df['statement'].apply(clean_text)
        df['prediction'] = df['cleaned_statement'].apply(
            lambda x: label[loaded_pipeline.predict([x])[0]]
        )

        results = list(zip(df['statement'], df['prediction']))

        prediction_counts = df['prediction'].value_counts().to_dict()

        Positive = prediction_counts.get('Normal', 0)/sum(prediction_counts.values()) *100
        Negative = 100 - Positive

        wordcloud_image = generate_wordcloud(' '.join(df['cleaned_statement']))

    else:
        Positive = 50
        Negative = 50


    # Prepare data for the template
    context = {
        'results': results,  # Pass statement-prediction pairs
        'growth_data': Positive if Positive > Negative else Negative,
        'Positive': Positive,
        'Negative': Negative,
        'type': type,
        'prediction_counts': prediction_counts,
        'wordcloud_image': wordcloud_image
    }

    return render_template('main.html', **context)

if __name__ == '__main__':
    app.run(debug=True)






