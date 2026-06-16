import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt

# Load dataset with full path
try:
    df = pd.read_csv(r"C:\Users\HP\Desktop\codveda-tasks\3) Sentiment dataset.csv")
    print("Dataset loaded successfully!\n")
    print(df.head())
except FileNotFoundError:
    print("Error: CSV file not found. Please check the path.")
    exit()

# Function to classify sentiment
def get_sentiment(text):
    polarity = TextBlob(str(text)).sentiment.polarity

    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

# Apply sentiment analysis (handle missing values safely)
df["Sentiment"] = df["Text"].fillna("").apply(get_sentiment)

# Show results
print("\nSentiment Counts:")
print(df["Sentiment"].value_counts())

# Create bar chart
sentiment_counts = df["Sentiment"].value_counts()

plt.figure(figsize=(6, 4))
sentiment_counts.plot(kind="bar", color=["green", "red", "gray"])
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.tight_layout()

# Show chart
plt.show()

# Save chart as image
plt.savefig("sentiment_chart.png")

# Save results to CSV
df.to_csv("sentiment_results.csv", index=False)

print("\nAnalysis complete!")
print("Results saved as sentiment_results.csv")
print("Chart saved as sentiment_chart.png")