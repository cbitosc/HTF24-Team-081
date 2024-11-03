

import nltk
from flask import Flask, request, render_template
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('vader_lexicon')

app = Flask(__name__)
sia = SentimentIntensityAnalyzer()

def is_fake_review(review):
    # Check sentiment score
    score = sia.polarity_scores(review)
    # Adjust thresholds for detecting strong sentiments
    if score['compound'] >= 0.5 or score['compound'] <= -0.5:  # Strong sentiment
        return True
    
    # Check length of the review
    if len(review.split()) < 5:  # Very short review
        return True
    
    return False

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    review = request.form.get('review')
    if review:
        score = sia.polarity_scores(review)
        sentiment = 'Positive' if score['compound'] > 0.05 else 'Negative' if score['compound'] < -0.05 else 'Neutral'
        fake_detection = is_fake_review(review)  # Check if review is potentially fake
        return render_template('index.html', result=(review, sentiment, score, fake_detection))
    
    return render_template('index.html', error="Please enter a review.")

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return render_template('index.html', error="No file uploaded.")
    
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error="No file selected.")
    
    reviews = file.read().decode('utf-8').splitlines()
    results = []
    for review in reviews:
        score = sia.polarity_scores(review)
        sentiment = 'Positive' if score['compound'] > 0.05 else 'Negative' if score['compound'] < -0.05 else 'Neutral'
        fake_detection = is_fake_review(review)  # Check if review is potentially fake
        results.append((review, sentiment, score, fake_detection))
    
    return render_template('index.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)
