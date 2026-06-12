from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

nltk.download("vader_lexicon")

sia = SentimentIntensityAnalyzer()

text = "Apple stock shows strong growth this quarter."

score = sia.polarity_scores(text)

print(score)