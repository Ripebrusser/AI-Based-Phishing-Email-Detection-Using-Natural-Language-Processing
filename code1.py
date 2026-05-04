import pandas as pd
import zipfile

# unzip
with zipfile.ZipFile("CEAS_08.csv.zip", 'r') as zip_ref:
    zip_ref.extractall()

# load
df = pd.read_csv("CEAS_08.csv")

# check columns
print(df.columns)

# FIXED columns
X = df['body'].astype(str)
y = df['label']

# TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(X)

# split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# predict
from sklearn.metrics import accuracy_score
pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))
