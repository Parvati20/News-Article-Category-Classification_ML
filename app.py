# from flask import Flask, render_template, request
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.naive_bayes import MultinomialNB

# app = Flask(__name__)

# df = pd.read_json("News_Category_Dataset_v3.json", lines=True)
# df = df[['category', 'headline', 'short_description']]
# df['text'] = df['headline'] + " " + df['short_description']
# df = df[['category', 'text']]

# X = df['text']
# y = df['category']

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
# X_train_vec = vectorizer.fit_transform(X_train)

# model = MultinomialNB()
# model.fit(X_train_vec, y_train)

# print(" Model trained and ready")

# @app.route("/", methods=["GET", "POST"])
# def home():
#     prediction = ""
#     if request.method == "POST":
#         news = request.form["news"]
#         vec = vectorizer.transform([news])
#         prediction = model.predict(vec)[0]

#     return render_template("index.html", result=prediction)

# if __name__ == "__main__":
#     app.run(debug=True)

from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

print("Model loaded successfully")

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = ""
    if request.method == "POST":
        news = request.form["news"]
        vec = vectorizer.transform([news])
        prediction = model.predict(vec)[0]

    return render_template("index.html", result=prediction)

if __name__ == "__main__":
    app.run(debug=True)
