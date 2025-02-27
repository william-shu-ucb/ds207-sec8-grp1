# %% [markdown]
# # Sentiment Analysis on Movie Reviews - Baseline Model
# 
# ## Introduction
# This notebook serves as a baseline model for the **Sentiment Analysis on Movie Reviews** competition. The goal is to classify movie reviews as different sentiment classes.
# 
# In this notebook, we will:
# 1. **Load and explore the dataset**
# 2. **Preprocess the text (cleaning, tokenization, TF-IDF)**
# 3. **Train a baseline model (Linear Regression, Logistic Regression, and NN)**
# 4. **Evaluate its performance**
# 

# %% [markdown]
# ### Import Libraries

# %%
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import string

# %% [markdown]
# ### Step 1: Data Ingestion

# %%
df = pd.read_csv("train.tsv", sep="\t")
df.head()

# %%
df = df[['PhraseId', 'Phrase','Sentiment']]
df.shape

# %%
unique_sentiment_percent = df["Sentiment"].value_counts().reset_index().sort_values(by='Sentiment')
unique_sentiment_percent.columns = ['Sentiment', 'count']
unique_sentiment_percent['proportion'] = unique_sentiment_percent['count'] / df.shape[0] * 100
unique_sentiment_percent['proportion'] = unique_sentiment_percent['proportion'].apply(lambda x: f"{x:.2f}%")
unique_sentiment_percent

# %%
sns.countplot(x=df["Sentiment"], palette="viridis")
plt.title("Sentiment Class Distribution")
plt.xlabel("Sentiment Class")
plt.ylabel("Count")
plt.show()


# %%
df["Phrase_length"] = df["Phrase"].apply(lambda x: len(x.split()))
print(df["Phrase_length"].describe())

plt.figure(figsize=(10,5))
sns.histplot(df["Phrase_length"], bins=30, kde=True)
plt.title("Distribution of Phrase Lengths")
plt.xlabel("Number of Words in Phrase")
plt.ylabel("Frequency")
plt.show()


# %%
word_counts = Counter(" ".join(df["Phrase"]).split())
common_words = word_counts.most_common(50)
plt.figure(figsize=(12,10))
sns.barplot(x=[word[1] for word in common_words], y=[word[0] for word in common_words])
plt.title("Most Common Words in Dataset")
plt.xlabel("Frequency")
plt.ylabel("Words")
plt.show()


# %% [markdown]
# ---
# ### Step 2: Data preprocessing

# %%
# Shuffle the dataset
df = df.sample(frac=1, random_state=1234).reset_index(drop=True)
# Lowercasing
df["Phrase"] = df["Phrase"].str.lower()
# Remove Punctuation
df["Phrase"] = df["Phrase"].str.translate(str.maketrans("", "", string.punctuation))
# Remove Stopwords
stop_words = set(stopwords.words("english"))
df["Phrase"] = df["Phrase"].apply(lambda text: " ".join([word for word in text.split() if word not in stop_words]))
# Lemmatization
lemmatizer = WordNetLemmatizer()
df["Phrase"] = df["Phrase"].apply(lambda text: " ".join([lemmatizer.lemmatize(word, pos="v") for word in text.split()]))

# %% [markdown]
# #### TF-IDF

# %%
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X = vectorizer.fit_transform(df["Phrase"])
y = df["Sentiment"]
print(f"TF-IDF feature matrix shape: {X.shape}")

# %%
feature_names = vectorizer.get_feature_names_out()
first_text_vector = X[0].toarray().flatten()
# see TF-IDF words
top_n = 10
top_indices = np.argsort(first_text_vector)[::-1][:top_n]
top_words = [feature_names[i] for i in top_indices]
top_scores = [first_text_vector[i] for i in top_indices]
print("Top TF-IDF words in first text:")
for word, score in zip(top_words, top_scores):
    print(f"{word}: {score:.4f}")

# %% [markdown]
# #### Train and Test Split

# %%
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=1234)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=1234)

print(f"Train set: {X_train.shape}, Validation set: {X_val.shape}, Test set: {X_test.shape}")

# %% [markdown]
# ---
# ### Step 3: Exploratory data analysis (EDA)

# %%


# %% [markdown]
# ---
# ### Step 4: BaseLine Models

# %%


# %% [markdown]
# ---
# ### Step 5: Evaluation

# %%



