# Customer Sentiment Analysis System
# Developed by Yamini TR | Using Python, NLP, and Scikit-learn

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Sample dataset (you can replace with your own CSV)
data = {
    'review': [
        "The product is great, I love it!",
        "Worst experience ever, not worth it.",
        "It's okay, not too bad but could be better.",
        "Amazing quality and fast delivery!",
        "Terrible product, completely disappointed."
    ],
    'sentiment': ['positive', 'negative', 'neutral', 'positive', 'negative']
}

df = pd.DataFrame(data)

# Step 2: Train-test split
X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size=0.3, random_state=42)

# Step 3: Text vectorization (TF-IDF)
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Step 4: Model training
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Step 5: Prediction and evaluation
y_pred = model.predict(X_test_tfidf)
print("✅ Model Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

# Step 6: Try new reviews
new_reviews = [
    "I love this product, very useful!",
    "Totally waste of money.",
    "Average experience, nothing special."
]

new_tfidf = vectorizer.transform(new_reviews)
predictions = model.predict(new_tfidf)

print("\n--- New Predictions ---")
for review, sentiment in zip(new_reviews, predictions):
    print(f"Review: {review}\nPredicted Sentiment: {sentiment}\n")
