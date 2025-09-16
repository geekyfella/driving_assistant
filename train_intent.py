# train_intent.py
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

df = pd.read_csv("intents.csv")
X = df['utterance'].astype(str)
y = df['intent'].astype(str)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_features=2000)),
    ("clf", LogisticRegression(max_iter=1000, C=1.0))
])

pipeline.fit(X_train, y_train)

# eval
y_pred = pipeline.predict(X_test)
print("ACC:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# save
joblib.dump(pipeline, "intent_clf.pkl")
print("Saved intent_clf.pkl")
