# %% [markdown]
# # Sentiment Analysis on Movie Reviews - Baseline Model
# 
# **Author**: Nicole Zhang, Viola Qiu, William Shu
# 
# ## Introduction
# This notebook serves as a baseline model for the **Sentiment Analysis on Movie Reviews** competition hosted on Kaggle ([competition link](https://www.kaggle.com/competitions/sentiment-analysis-on-movie-reviews/overview)). The objective of this competition is to classify movie reviews into five sentiment categories (0: negative, 1: somewhat negative, 2: neutral, 3: somewhat positive, 4: positive) based on textual phrases extracted from Rotten Tomatoes reviews. The dataset contains phrases labeled with sentiment scores, making it a multi-class text classification task.
# 
# In this notebook, we will:
# 1. **Load and explore the dataset**
# 2. **Preprocess the text (cleaning, tokenization, TF-IDF)**
# 3. **Train baseline models (Logistic Regression, Random Forest, XGBoost, LightGBM, and Neural Network)**
# 4. **Select the best model based on their performance**
# 5. **Tuning hyperparamter of the selected model**
# 6. **Discuss next step plan**
# 
# 

# %% [markdown]
# ### Import Libraries

# %%
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize  # Corrected from 'atokenize'
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import string
import xgboost as xgb
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
# New imports for embeddings and CNN
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import ParameterSampler
nltk.download('punkt', quiet=True)

# %% [markdown]
# ### Step 1: Data Ingestion
# 
# In this step, we load the `train.tsv` file into a Pandas DataFrame and perform basic checks. This includes verifying the overall structure of the dataset (rows and columns) and previewing a few entries with `df.head()`. These actions ensure the data is properly loaded before we proceed with further exploration and analysis.

# %%
df = pd.read_csv("train.tsv", sep="\t")
df.head()

# %% [markdown]
# #### Conclusion
# - The dataset contains columns: `PhraseId`, `SentenceId`, `Phrase`, and `Sentiment`.
# - Initial rows show phrases of varying lengths (from full sentences to single words) with sentiment labels ranging from 0 to 4.

# %% [markdown]
# ---
# ### Step 2: Data Cleaning
# 
# In this step, we ensure the dataset is properly prepared before moving on to deeper analysis and modeling. We also wrap the cleaning steps in a function to be used for later loading test data and do the same cleaning.
# 1. **Checking for missing values** and removing or correcting any invalid rows (for example, those with empty `Phrase` fields).
# 2. **Remove the `SentenceId` column**, because it functions purely as an identifier for grouping phrases by sentence and does not provide direct value for our initial baseline model.
# 3. **Remove rows** where phrase is empty.
# 4. **Check for duplicates** and address any outliers, ensuring our data is clean and consistent for the subsequent steps.
# 5. **Lower Case** to standardizes the text, so that words like "Movie" and "movie" are treated as the same token.
# 6. **Remove Punctuation**: Punctuation is typically not useful for sentiment classification and can add noise, so removing it simplifies the text.
# 7. **Lemmatization**: Normalize different variations of the same word (e.g., “running” → “run”).

# %%
def clean_data(df):
    # 1. Check for missing values
    print("Missing values in each column:")
    print(df.isnull().sum())

    # 2. Remove the SentenceId column
    #    This column is not needed for our baseline model.
    df = df[['PhraseId', 'Phrase','Sentiment']]
    df.shape

    # 3. Reemove rows where phrase is empty.
    initial_count = df.shape[0]
    df = df[df['Phrase'].str.strip() != '']
    removed_empty_count = initial_count - df.shape[0]
    print(f"\nRemoved {removed_empty_count} rows where Phrase was empty.")

    # 4. Check for duplicates
    duplicates_count = df.duplicated().sum()
    df.drop_duplicates(inplace=True)
    print(f"\nNumber of duplicate rows: {duplicates_count}")

    # 5. Lowercasing
    df["Phrase"] = df["Phrase"].str.lower()

    # 6. Remove Punctuation
    df["Phrase"] = df["Phrase"].str.translate(str.maketrans("", "", string.punctuation))

    # 7. Lemmatization
    nltk.download('wordnet', quiet=True)
    lemmatizer = WordNetLemmatizer()
    df["Phrase"] = df["Phrase"].apply(lambda text: " ".join([lemmatizer.lemmatize(word, pos="v") for word in text.split()]))

clean_data(df=df)

# %%
# Alternative cleaning for embeddings
def clean_data_for_embeddings(df):
    print("Cleaning data for embeddings...")
    df = df[['PhraseId', 'Phrase', 'Sentiment']]
    initial_count = df.shape[0]
    df = df[df['Phrase'].str.strip() != '']
    print(f"Removed {initial_count - df.shape[0]} rows where Phrase was empty.")
    df['Phrase'] = df['Phrase'].str.lower()  # Keep lowercase, skip punctuation removal (punctuation could emphasize the sentiment) and lemmatization (could have its own nuances and GloVe already pre-trained with lemmatization)
    return df

# Create a separate DataFrame for embeddings
df_emb = clean_data_for_embeddings(df.copy())

# %% [markdown]
# **Purpose**
# - Cleaning ensures the data is consistent and free of noise, improving model performance and reproducibility (especially for test data).
# - Removing `SentenceId` simplifies the feature set for the baseline, as it’s not directly relevant to sentiment.
# - Text normalization (lowercase, punctuation removal, lemmatization) reduces variability and prepares the text for feature extraction.
# 
# **Conclusion**  
# - No missing values were present in the dataset.
# - Removed 1 empty phrase row and found no duplicates.

# %% [markdown]
# ---
# ### Step 3: Exploratory data analysis
# 
# #### Sentiment Distribution
# 
# **Purpose**  
# Understanding sentiment distribution is important because it reveals class imbalances that could bias the model's learning process. A heavily skewed distribution might mean the model will have difficulty learning the minority classes. By examining the count and percentage of reviews in each sentiment category, we can determine if the dataset is balanced or if we need to apply techniques to address any imbalance.
# 
# 

# %%
# Calculate the count and percentage of each sentiment class
sentiment_percent = df["Sentiment"].value_counts().reset_index().sort_values(by='Sentiment')
sentiment_percent.columns = ['Sentiment', 'count']
sentiment_percent['proportion'] = sentiment_percent['count'] / df.shape[0] * 100
sentiment_percent['proportion'] = sentiment_percent['proportion'].apply(lambda x: f"{x:.2f}%")

print("Sentiment Distribution (% of total reviews):")
print(sentiment_percent.to_string(index=False))

# Visualize the sentiment distribution
plt.figure(figsize=(7, 4))
sns.countplot(data=df, x="Sentiment", hue="Sentiment", palette="viridis")
plt.title('Sentiment Class Distribution', fontsize=14)
plt.xlabel('Sentiment Class', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.tight_layout()
plt.show()

# %% [markdown]
# **Conclusion**  
# From the chart, it appears that **neutral (sentiment 2)** dominates the dataset, while **extremely positive (4)** and **extremely negative (0)** are less frequent. Because of this imbalance, we may need to take extra steps to prevent the model from focusing too heavily on the most common sentiment.

# %% [markdown]
# ---
# ### Step 3: Exploratory Data Analysis - cont'd
# 
# #### Phrase Length Analysis
# 
# **Purpose**  
# Examining the length of each phrase (in words) helps us determine whether certain sentiments are associated with shorter or longer text. It also reveals potential outliers (very short or very long phrases) that may skew the model’s performance.

# %%
df["Phrase_length"] = df["Phrase"].apply(lambda x: len(x.split()))
plt.figure(figsize=(10,6))
sns.boxplot(x='Sentiment', y='Phrase_length', data=df, hue='Sentiment', palette="viridis")
plt.title("Phrase Length Distribution by Sentiment")
plt.xlabel("Sentiment")
plt.ylabel("Phrase Length (word count)")
plt.legend('', frameon=False)
plt.show()

# %% [markdown]
# **Conclusion**  
# From the boxplot, we see that **neutral** (sentiment 2) phrases tend to be shorter, while other sentiments have higher median lengths and more outliers. This pattern suggests that expressing a stronger sentiment—whether positive or negative—often involves using more words.

# %% [markdown]
# ---
# ### Step 3: Exploratory Data Analysis - cont'd
# 
# #### Most Common Words by Sentiment
# 
# **Purpose**  
# By identifying the top words for each sentiment category, we can see how language differs across very negative (0), negative (1), neutral (2), positive (3), and very positive (4) phrases. This insight helps us understand which words are common filler terms versus which words might be truly indicative of a certain sentiment.

# %%
fig, axes = plt.subplots(1, 5, figsize=(20, 8))
sentiments = sorted(df['Sentiment'].unique())
for i, sentiment in enumerate(sentiments):
    subset = df[df['Sentiment'] == sentiment]
    word_counts = Counter(" ".join(subset["Phrase"]).split())
    common_words = word_counts.most_common(15)
    sns.barplot(ax=axes[i],
                x=[count for _, count in common_words],
                y=[word for word, _ in common_words],
                orient='h', color='#4c72b0')
    axes[i].set_title(f"Sentiment {sentiment}")
    axes[i].set_xlabel("Frequency")
    axes[i].set_ylabel("Word")

plt.tight_layout()
plt.show()


# %% [markdown]
# **Conclusion**  
# We observe that some words (e.g., "the", "and") appear frequently in all sentiments, suggesting they are general filler words. Meanwhile, other words—though not immediately obvious from this particular top-10 list—may be more sentiment-specific and could help differentiate negative from positive phrases. These findings can guide us in refining our stopwords list or engineering sentiment-focused features for more accurate classification.

# %% [markdown]
# ---
# ### Step 4: Data Preprocessing
# 
# In this step, we standardize and transform the text data to prepare it for modeling.
#    
# 1. **Stopword Removal:**  
#    We define a set of English stopwords (using NLTK) to filter out common words (like "the", "and", etc.) that generally don't contribute to sentiment detection. We manually set a set of important words to exclude from predefined NLTK stop word list.
# 
# 2. **TF-IDF Vectorization:**  
#    We convert the cleaned text into numerical features using TF-IDF, which captures both the term frequency and the relative importance of words across all documents.  
#    The TF-IDF vectorizer is fitted on the training set and then used to transform the test set, ensuring that the feature representation is consistent and avoids data leakage.
# 

# %%
# Download NLTK stopwords
nltk.download('stopwords', quiet=True)

# Load default stopwords
stop_words = set(stopwords.words("english"))

# Add custom stopwords
additional_stopwords = {"example", "anotherword"}
stop_words.update(additional_stopwords)

# Remove important words (negations etc.)
important_words = {
    "not", "no", "never", "none", "nor", "nothing", "nowhere",
    "hardly", "barely", "scarcely", "very", "too", "so", "just",
    "quite", "rather", "almost", "much",
    "aren't", "couldn't", "didn't", "doesn't", "hadn't", "hasn't",
    "haven't", "isn't", "mightn't", "mustn't", "needn't", "shan't",
    "shouldn't", "wasn't", "weren't", "won't", "wouldn't"
}
stop_words = stop_words - important_words

# Convert set to list
stop_words = list(stop_words)

# Initialize the TF-IDF Vectorizer with the custom stopwords list
vectorizer = TfidfVectorizer(
    stop_words=stop_words,
    max_features=5000
)

# Apply TF-IDF
X = vectorizer.fit_transform(df["Phrase"])
y = df["Sentiment"]

# Shape check
print(f"TF-IDF feature matrix shape: {X.shape}")

# Analyze top words for first sample
feature_names = vectorizer.get_feature_names_out()
first_text_vector = X[0].toarray().flatten()

# Top 10 words
top_n = 10
top_indices = np.argsort(first_text_vector)[::-1][:top_n]
top_words = [feature_names[i] for i in top_indices]
top_scores = [first_text_vector[i] for i in top_indices]

print("Top TF-IDF words in first text:")
for word, score in zip(top_words, top_scores):
    print(f"{word}: {score:.4f}")

# %%
max_len = 50
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df_emb['Phrase'])

X_seq = tokenizer.texts_to_sequences(df_emb['Phrase'])

X_pad = pad_sequences(X_seq, maxlen=max_len, padding='post', truncating='post')
y_emb = df_emb['Sentiment']

X_train_pad, X_temp_pad, y_train_emb, y_temp_emb = train_test_split(X_pad, y_emb, test_size=0.4, random_state=1234)
X_val_pad, X_test_pad, y_val_emb, y_test_emb = train_test_split(X_temp_pad, y_temp_emb, test_size=0.5, random_state=1234)

print(f"Shape of padded training data: {X_train_pad.shape}")
print(f"Vocabulary size: {len(tokenizer.word_index) + 1}")

# %% [markdown]
# We use GloVe embeddings to capture semantic relationships in the text. The embedding matrix maps each word in our vocabulary to its pre-trained vector.

# %%
def load_glove_embeddings(glove_file_path, word_index, embedding_dim=100):
    embeddings_index = {}
    with open(glove_file_path, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    
    vocab_size = len(word_index) + 1
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    
    return embedding_matrix

glove_path = 'glove.6B.100d.txt'
embedding_dim = 100
embedding_matrix = load_glove_embeddings(glove_file_path=glove_path, 
                                        word_index=tokenizer.word_index, 
                                        embedding_dim=embedding_dim)

print(f"Embedding matrix shape: {embedding_matrix.shape}")

# %% [markdown]
# #### Train, Validation, Test Split
# 1. **Split into Train and Validation Sets**: Divide the training data (from `train.tsv`) into 60% for training, 20% for validation, 20% for tests.

# %%
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=1234)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=1234)

print(f"Train set: {X_train.shape}, Validation set: {X_val.shape}, Test set: {X_test.shape}")

# %% [markdown]
# **Conclusion**
# - The split resulted in approximately 93,636 training samples, 31,212 validation samples and 31,211 test samples.

# %% [markdown]
# ---
# ### Step 5: BaseLine Models

# %% [markdown]
# #### Step 5-0: CNN with GloVe Embeddings
# 
# Convolutional Neural Network to capture sentiment-indicating phrases using GloVe embeddings. The model uses multiple kernel sizes to detect patterns of varying lengths and includes global pooling for robust feature extraction.

# %%
def build_cnn_model(vocab_size, embedding_dim, max_len, embedding_matrix, kernel_sizes=[3, 5], pooling='max', trainable_embeddings=False):
    inputs = keras.layers.Input(shape=(max_len,))
    embedding = keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        weights=[embedding_matrix],
        input_length=max_len,
        trainable=trainable_embeddings
    )(inputs)
    
    conv_layers = []
    for ks in kernel_sizes:
        conv = keras.layers.Conv1D(filters=64, kernel_size=ks, activation='relu', padding='same')(embedding)
        if pooling == 'max':
            pooled = keras.layers.MaxPooling1D(pool_size=2)(conv)
        elif pooling == 'avg':
            pooled = keras.layers.AveragePooling1D(pool_size=2)(conv)
        conv_layers.append(pooled)
    
    if len(kernel_sizes) > 1:
        concat = keras.layers.Concatenate()(conv_layers)
    else:
        concat = conv_layers[0]
    
    if pooling == 'max':
        pooled = keras.layers.GlobalMaxPooling1D()(concat)
    else:
        pooled = keras.layers.GlobalAveragePooling1D()(concat)
    
    dense = keras.layers.Dense(128, activation='relu')(pooled)
    dropout = keras.layers.Dropout(0.5)(dense)
    outputs = keras.layers.Dense(5, activation='softmax')(dropout)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

vocab_size = len(tokenizer.word_index) + 1

cnn_model = build_cnn_model(vocab_size, embedding_dim, max_len, embedding_matrix)
cnn_model.summary()

history = cnn_model.fit(
    X_train_pad, y_train_emb,
    epochs=10,
    batch_size=32,
    validation_data=(X_val_pad, y_val_emb),
    verbose=1
)

y_pred_cnn = np.argmax(cnn_model.predict(X_test_pad), axis=1)
accuracy_cnn = accuracy_score(y_test_emb, y_pred_cnn)
precision_cnn = precision_score(y_test_emb, y_pred_cnn, average='weighted')
recall_cnn = recall_score(y_test_emb, y_pred_cnn, average='weighted')
f1_cnn = f1_score(y_test_emb, y_pred_cnn, average='weighted')
report_cnn = classification_report(y_test_emb, y_pred_cnn)

print("\nCNN with GloVe Embeddings - Evaluation Metrics")
print(f"Accuracy: {accuracy_cnn:.4f}")
print(f"Precision (Weighted): {precision_cnn:.4f}")
print(f"Recall (Weighted): {recall_cnn:.4f}")
print(f"F1 Score (Weighted): {f1_cnn:.4f}")
print("\nClassification Report:")
print(report_cnn)

# Plot confusion matrix
conf_matrix_cnn = confusion_matrix(y_test_emb, y_pred_cnn)
plt.figure(figsize=(8,6))
sns.heatmap(conf_matrix_cnn, annot=True, fmt='d', cmap='Blues',
            xticklabels=[0,1,2,3,4],
            yticklabels=[0,1,2,3,4])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - CNN with GloVe")
plt.show()

# %% [markdown]
# We test GloVe embeddings with other dimensions, 50D, 100D, and 200D to evaluate their impact on CNN performance.

# %%
embedding_dims = [50, 100, 200, 300]
results_emb = []

for dim in embedding_dims:
    print(f"\nTraining CNN with {dim}D GloVe embeddings...")
    glove_path = f'glove.6B.{dim}d.txt'
    embedding_matrix = load_glove_embeddings(glove_path, tokenizer.word_index, dim)
    cnn_model = build_cnn_model(vocab_size, dim, max_len, embedding_matrix)
    cnn_model.fit(X_train_pad, y_train_emb, epochs=10, batch_size=32, validation_data=(X_val_pad, y_val_emb), verbose=0)
    y_pred = np.argmax(cnn_model.predict(X_test_pad), axis=1)
    results_emb.append({
        'embedding_dim': dim,
        'accuracy': accuracy_score(y_test_emb, y_pred),
        'precision': precision_score(y_test_emb, y_pred, average='weighted'),
        'recall': recall_score(y_test_emb, y_pred, average='weighted'),
        'f1': f1_score(y_test_emb, y_pred, average='weighted')
    })

print("\nEmbedding Dimension Experiment Results:")
print(pd.DataFrame(results_emb))

# %% [markdown]
# Try with different kernel_sizes.

# %%
configs = [
    {'kernel_sizes': [3], 'pooling': 'max'},
    {'kernel_sizes': [3, 5], 'pooling': 'max'},
    {'kernel_sizes': [3, 5, 7], 'pooling': 'avg'}
]

embedding_dim = 300
vocab_size = len(tokenizer.word_index) + 1

glove_path = 'glove.6B.300d.txt'
embedding_matrix = load_glove_embeddings(glove_file_path=glove_path, 
                                        word_index=tokenizer.word_index, 
                                        embedding_dim=embedding_dim)

results_cnn = []
for config in configs:
    print(f"\nTraining CNN with kernel_sizes={config['kernel_sizes']}, pooling={config['pooling']}...")
    cnn_model = build_cnn_model(vocab_size, embedding_dim, max_len, embedding_matrix, **config)
    cnn_model.fit(X_train_pad, y_train_emb, epochs=10, batch_size=32, 
                  validation_data=(X_val_pad, y_val_emb), verbose=0)
    y_pred = np.argmax(cnn_model.predict(X_test_pad), axis=1)
    results_cnn.append({
        'kernel_sizes': config['kernel_sizes'],
        'pooling': config['pooling'],
        'accuracy': accuracy_score(y_test_emb, y_pred),
        'precision': precision_score(y_test_emb, y_pred, average='weighted'),
        'recall': recall_score(y_test_emb, y_pred, average='weighted'),
        'f1': f1_score(y_test_emb, y_pred, average='weighted')
    })

print("\nCNN Architecture Experiment Results:")
print(pd.DataFrame(results_cnn))

# %% [markdown]
# Kernel size 3 performed the best with 65.19% accuracy

# %% [markdown]
# Hyper parameter tunning.

# %% [markdown]
# #### 5-1. Logistic Regression Classification model
# Logistic Regression is a simple, interpretable linear model, making it an ideal baseline for multi-class classification tasks.

# %%
# Train the Logistic Regression classification model

model_logistic = LogisticRegression(max_iter=500)
model_logistic.fit(X_train, y_train)

y_pred_logistic = model_logistic.predict(X_test)
acc_logistic = accuracy_score(y_test, y_pred_logistic)

# Evaluation Metrics
accuracy_LR = accuracy_score(y_test, y_pred_logistic)
precision_LR = precision_score(y_test, y_pred_logistic, average='weighted')
recall_LR = recall_score(y_test, y_pred_logistic, average='weighted')
f1_LR = f1_score(y_test, y_pred_logistic, average='weighted')
report_LR = classification_report(y_test, y_pred_logistic)

# Print Summary
print("Logistic Regression - Evaluation Metrics")
print(f"Accuracy (LR): {accuracy_LR:.4f}")
print(f"Precision (Weighted) (LR): {precision_LR:.4f}")
print(f"Recall (Weighted) (LR): {recall_LR:.4f}")
print(f"F1 Score (Weighted) (LR): {f1_LR:.4f}")
print("\nClassification Report (LR):")
print(report_LR)

# Confusion Matrix
conf_matrix_LR  = confusion_matrix(y_test, y_pred_logistic)

# Plot Confusion Matrix
plt.figure(figsize=(8,6))
sns.heatmap(conf_matrix_LR, annot=True, fmt='d', cmap='Blues',
            xticklabels=[0,1,2,3,4],
            yticklabels=[0,1,2,3,4])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - Logistic Regression")
plt.show()


# %% [markdown]
# **Conclusion**
# - Logistic Regression achieved an accuracy of 62.26%, with a weighted F1-score of 0.5928.
# - High recall for neutral class (2) but lower performance on minority classes (0 and 4).

# %% [markdown]
# #### 5-2. Random Forest Model
# Random Forest captures non-linear relationships and interactions between features, potentially improving over Logistic Regression.

# %%
# Initialize Random Forest model

model_RF = RandomForestClassifier(n_estimators=100, random_state=1234, n_jobs=10)
model_RF.fit(X_train, y_train)

# Predict on test data
y_pred = model_RF.predict(X_test)

# Fit model on training data
model_RF.fit(X_train, y_train)

# Predict on test data
y_pred = model_RF.predict(X_test)

# Evaluate
acc_RF= accuracy_score(y_test, y_pred)
print(f"Random Forest Accuracy: {acc_RF:.4f}")

# Print classification report
print(classification_report(y_test, y_pred))

# %% [markdown]
# **Conclusion**
# - Accuracy of 62.93% and F1-score of 0.62, a slight improvement over Logistic Regression.
# - Still biased toward the neutral class, indicating limited ability to handle imbalance without tuning.

# %% [markdown]
# #### 5-3. XGBoost Classification Model
# XGBoost is a powerful gradient-boosting algorithm, often excelling in structured data tasks.
# 

# %%
# Train the XGBoost classification model
model_xg = xgb.XGBClassifier(
    objective="multi:softmax",  # Multi-class classification task
    num_class=5,  # 5 sentiment classes (0-4)
    max_depth=10,  # Maximum tree depth
    learning_rate=0.1,  # Learning rate
    n_estimators=500,  # Number of trees
    eval_metric="mlogloss",  # Multi-class loss function
    tree_method="gpu_hist"
)
model_xg.fit(X_train, y_train)

# Make predictions
y_pred_xg  = model_xg.predict(X_test)

# Evaluate and Calculate accuracy
accuracy_xg = accuracy_score(y_test, y_pred_xg)
precision_xg = precision_score(y_test, y_pred_xg, average='weighted')
recall_xg = recall_score(y_test, y_pred_xg, average='weighted')
f1_xg = f1_score(y_test, y_pred_xg, average='weighted')
report_xg = classification_report(y_test, y_pred_xg)

# Print Summary
print("\nXGBoost - Evaluation Metrics")
print(f"Accuracy: {accuracy_xg:.4f}")
print(f"Precision (Weighted): {precision_xg:.4f}")
print(f"Recall (Weighted): {recall_xg:.4f}")
print(f"F1 Score (Weighted): {f1_xg:.4f}")
print("\nClassification Report:")
print(report_xg)

# Confusion Matrix
conf_matrix_xg = confusion_matrix(y_test, y_pred_xg)

# Plot Confusion Matrix
plt.figure(figsize=(8,6))
sns.heatmap(conf_matrix_xg, annot=True, fmt='d', cmap='Blues',
            xticklabels=[0,1,2,3,4],
            yticklabels=[0,1,2,3,4])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - XGBoost")
plt.show()

# %% [markdown]
# **Conclusion**
# - Poor performance: accuracy of 52.04% and F1-score of 0.3881.

# %% [markdown]
# #### 5-4. LightGBM Classification Model
# LightGBM is optimized for efficiency with high-dimensional sparse data, potentially outperforming XGBoost.

# %%
# Initialize LightGBM model
model_gbm = LGBMClassifier(
    n_estimators=200,
    num_leaves=63,
    learning_rate=0.1,
    n_jobs=-1,
    random_state=1234,
)
model_gbm.fit(X_train, y_train)

# Predict
y_pred_gbm = model_gbm.predict(X_test)

# Evaluation Metrics
accuracy_LGBM = accuracy_score(y_test, y_pred_gbm)
precision_LGBM = precision_score(y_test, y_pred_gbm, average='weighted')
recall_LGBM = recall_score(y_test, y_pred_gbm, average='weighted')
f1_LGBM = f1_score(y_test, y_pred_gbm, average='weighted')
report_LGBM = classification_report(y_test, y_pred_gbm)

# Print Summary
print("LightGBM - Evaluation Metrics")
print(f"Accuracy (LGBM): {accuracy_LGBM:.4f}")
print(f"Precision (Weighted) (LGBM): {precision_LGBM:.4f}")
print(f"Recall (Weighted) (LGBM): {recall_LGBM:.4f}")
print(f"F1 Score (Weighted) (LGBM): {f1_LGBM:.4f}")
print("\nClassification Report (LGBM):")
print(report_LGBM)

# Confusion Matrix
conf_matrix_LGBM = confusion_matrix(y_test, y_pred_gbm)

# Plot Confusion Matrix
plt.figure(figsize=(8,6))
sns.heatmap(conf_matrix_LGBM, annot=True, fmt='d', cmap='Blues',
            xticklabels=[0,1,2,3,4],
            yticklabels=[0,1,2,3,4])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - LightGBM")
plt.show()

# %% [markdown]
# **Conclusion**
# - Accuracy of 63.00% and F1-score of 0.6024, matching Random Forest but with faster training.
# - Better balanced performance across classes compared to XGBoost, though still favoring class 2.

# %% [markdown]
# #### 5-5. Neural Network model
# Neural networks can capture complex non-linear patterns in text data, potentially outperforming traditional models.

# %%
# Train Neural Network model
def build_model(n_classes=5,
                 input_dim=5000,
                 hidden_layer_sizes=[],
                 activation='relu',
                 optimizer='adam',
                 learning_rate=0.01,
                 metric='accuracy'):
    """
    Build a neural network model for multi-class sentiment classification.
    Args:
    - n_classes: Number of sentiment classes (5: 0-4).
    - input_dim: Input dimension (number of TF-IDF features).
    - hidden_layer_sizes: List of hidden layer sizes (e.g., [512, 256]).
    - activation: Activation function (e.g., 'relu', 'tanh').
    - optimizer: Optimizer to use (e.g., 'adam', 'sgd').
    - learning_rate: Learning rate for optimizer.
    - metric: Evaluation metric to track.

    Returns:
    - model: Compiled tf.keras model.
    """
    tf.keras.backend.clear_session()
    np.random.seed(0)
    tf.random.set_seed(0)

    # Start building model
    model_nn = keras.Sequential()

    # Input layer (TF-IDF size)
    model_nn.add(keras.layers.Input(shape=(input_dim,)))

    # Hidden layers
    for units in hidden_layer_sizes:
        model_nn.add(keras.layers.Dense(units, activation=activation))
        model_nn.add(keras.layers.Dropout(0.3))  # Optional regularization

    # Output layer (5 sentiment classes)
    model_nn.add(keras.layers.Dense(n_classes, activation='softmax'))

    # Select optimizer
    if optimizer.lower() == 'adam':
        optimizer_instance = keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer.lower() == 'sgd':
        optimizer_instance = keras.optimizers.SGD(learning_rate=learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer}")

    # Compile the model
    model_nn.compile(optimizer=optimizer_instance,
                  loss='categorical_crossentropy',  # Because of one-hot encoding
                  metrics=[metric])

    return model_nn



# %%
def train_and_evaluate_nn(hidden_layer_sizes=[512, 256],
                          activation='relu',
                          optimizer='adam',
                          learning_rate=0.01,
                          num_epochs=10,
                          batch_size=32):
    """
    Train and evaluate a Neural Network for sentiment analysis.
    All hyperparameters have default values but can be overridden.
    """

    print(f"\nTraining with config - hidden_layer_sizes: {hidden_layer_sizes}, "
          f"activation: {activation}, optimizer: {optimizer}, learning_rate: {learning_rate}")

    # Build model using the passed (or default) hyperparameters
    model = build_model(
        n_classes=5,
        input_dim=X_train.shape[1],
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        optimizer=optimizer,
        learning_rate=learning_rate
    )

    # One-hot encode labels
    y_train_nn = keras.utils.to_categorical(y_train, 5)
    y_test_nn = keras.utils.to_categorical(y_test, 5)

    # Train model
    history = model.fit(
        X_train.toarray(), y_train_nn,
        epochs=num_epochs,
        batch_size=batch_size,
        validation_data=(X_test.toarray(), y_test_nn),
        verbose=0
    )

    # Evaluate
    train_accuracy = history.history['accuracy'][-1]
    val_accuracy = history.history['val_accuracy'][-1]

    y_pred_nn = model.predict(X_test.toarray())
    y_pred_nn_labels = np.argmax(y_pred_nn, axis=1)
    y_test_labels = np.argmax(y_test_nn, axis=1)

    precision = precision_score(y_test_labels, y_pred_nn_labels, average='weighted')
    recall = recall_score(y_test_labels, y_pred_nn_labels, average='weighted')
    f1 = f1_score(y_test_labels, y_pred_nn_labels, average='weighted')

    # Summary
    print(f"Train Accuracy: {train_accuracy:.4f}, Val Accuracy: {val_accuracy:.4f}, F1: {f1:.4f}\n")

    # Return results as dictionary (for tracking in a list if needed)
    return {
        'hidden_layer_sizes': hidden_layer_sizes,
        'activation': activation,
        'optimizer': optimizer,
        'learning_rate': learning_rate,
        'train_accuracy': round(train_accuracy, 2),
        'val_accuracy': round(val_accuracy, 2),
        'precision': round(precision, 2),
        'recall': round(recall, 2),
        'f1': round(f1, 2)
    }

# %%
def train_and_evaluate_nn(hidden_layer_sizes=[512, 256],
                          activation='relu',
                          optimizer='adam',
                          learning_rate=0.01,
                          num_epochs=10,
                          batch_size=32):
    """
    Train and evaluate a Neural Network for sentiment analysis.
    All hyperparameters have default values but can be overridden.
    """

    print(f"\nTraining with config - hidden_layer_sizes: {hidden_layer_sizes}, "
          f"activation: {activation}, optimizer: {optimizer}, learning_rate: {learning_rate}")

    # Build model using the passed (or default) hyperparameters
    model = build_model(
        n_classes=5,
        input_dim=X_train.shape[1],
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        optimizer=optimizer,
        learning_rate=learning_rate
    )

    # One-hot encode labels
    y_train_nn = keras.utils.to_categorical(y_train, 5)
    y_test_nn = keras.utils.to_categorical(y_test, 5)

    # Train model
    history = model.fit(
        X_train.toarray(), y_train_nn,
        epochs=num_epochs,
        batch_size=batch_size,
        validation_data=(X_test.toarray(), y_test_nn),
        verbose=0
    )

    # Evaluate
    train_accuracy = history.history['accuracy'][-1]
    val_accuracy = history.history['val_accuracy'][-1]

    y_pred_nn = model.predict(X_test.toarray())
    y_pred_nn_labels = np.argmax(y_pred_nn, axis=1)
    y_test_labels = np.argmax(y_test_nn, axis=1)

    precision = precision_score(y_test_labels, y_pred_nn_labels, average='weighted')
    recall = recall_score(y_test_labels, y_pred_nn_labels, average='weighted')
    f1 = f1_score(y_test_labels, y_pred_nn_labels, average='weighted')

    # Summary
    print(f"Train Accuracy: {train_accuracy:.4f}, Val Accuracy: {val_accuracy:.4f}, F1: {f1:.4f}\n")

    # Return results as dictionary (for tracking in a list if needed)
    return {
        'hidden_layer_sizes': hidden_layer_sizes,
        'activation': activation,
        'optimizer': optimizer,
        'learning_rate': learning_rate,
        'train_accuracy': round(train_accuracy, 2),
        'val_accuracy': round(val_accuracy, 2),
        'precision': round(precision, 2),
        'recall': round(recall, 2),
        'f1': round(f1, 2)
    }


train_and_evaluate_nn()

# %% [markdown]
# **Conclusion**
# - Neural Network outperformed others (accuracy 64.14%, F1 0.6123).

# %% [markdown]
# ---
# ### Step 6: Model Evaluation and Selection
# 
# #### **6-1 Overview of Model Performance**
# We evaluated five different models for sentiment classification based on Accuracy, Weighted Precision, Weighted Recall, and Weighted F1-score.
# 
# 
# | **Model**                   | **Accuracy** | **Weighted Precision** | **Weighted Recall** | **Weighted F1-score** |
# |-----------------------------|------------|------------------------|---------------------|-----------------------|
# | **Logistic Regression**     | 0.6226     | 0.6037                 | 0.6226             | 0.5928               |
# | **Random Forest**           | 0.6293     | 0.6100                 | 0.6300             | 0.62                 |
# | **XGBoost**                 | 0.5204     | 0.4746                 | 0.5204             | 0.3881               |
# | **LightGBM**                | 0.6300     | 0.6099                 | 0.6300             | 0.6024               |
# | **Neural Network (MLP)**    | 0.6414     | 0.6300                 | 0.6400             | 0.6123               |
# | **CNN+Kenel(5)+Emb**        | 0.6499     | 0.6562                 | 0.6499             | 0.6452               |
# | **CNN+Kenel(3)+Emb**        | 0.6519     | 0.6470                 | 0.6519             | 0.6398               |
# 
# <br>
# 
# #### **6-2 Key Observations**
# ##### Logistic Regression
# - Accuracy: 62.26% – This serves as the simplest linear baseline.
# - High recall for class 2 (0.87), but lower recall for other classes, indicating a bias toward the majority class。
# - Fast computation but limited in capturing complex non-linear relationships.
# 
# ##### Random Forest
# - Accuracy: 62.93% – Slightly better than Logistic Regression.
# - Can handle non-linearity but does not significantly outperform Logistic Regression.
# 
# ##### XGBoost
# - Accuracy: 52.04% – The lowest-performing model.
# - Though XGBoost is powerful for structured data, it does not work well with TF-IDF features**.
# - Severely biased toward class 2 (recall = 0.97), resulting in poor recall for other classes**.
# 
# ##### LightGBM
# - Accuracy: 63.00% – The best performing tree-based model.
# - More efficient than XGBoost for high-dimensional sparse features like TF-IDF.
# - Performance is very close to Neural Network (MLP), but computationally cheaper.
# 
# ##### Neural Network (MLP)
# - Accuracy: 64.14% – The best-performing model so far.
# - Demonstrates better non-linear modeling capabilities compared to tree-based models.
# - Outperforms all other models by 1.5%-2%, making it a strong final model candidate.
# - Higher computational cost but justifies the performance gain.
# 
# <br>
# 
# #### **6-3 Final Model Selection**
# After evaluating several models, including **Logistic Regression**, **Random Forest**, **LightGBM**, and **XGBoost**,  
# we have selected the **Neural Network** as the best-performing model for the sentiment analysis task.
# 
# This decision is based on a comprehensive comparison of evaluation metrics such as accuracy, precision, recall,  
# F1-score, confusion matrices, and computational cost.
# 
# The Neural Network, with a baseline architecture of **[512, 256] hidden layers using ReLU activation and Adam optimizer**,  
# achieved the following performance:
# 
# - **Train Accuracy:** 0.7028
# - **Validation Accuracy:** 0.6414
# - **F1 Score:** 0.6123
# 
# These results demonstrate that the Neural Network, despite being more complex,  
# is better able to generalize to unseen data compared to simpler models.  
# Based on this, we have decided to focus the next stage of our work on **hyperparameter tuning for the Neural Network**,  
# aiming to further improve its generalization performance.
# 

# %% [markdown]
# ---
# ### Step 8: Hyperparameter Tunning

# %% [markdown]
# CNN Hyperparameter Tunning based on kernal size 3.

# %%
param_dist = {
    'filters': [64, 128],
    'dropout_rate': [0.3, 0.5],
    'learning_rate': [0.001, 0.01]
}
n_iter = 4
param_list = list(ParameterSampler(param_dist, n_iter=n_iter, random_state=1234))
results_tuning = []
for params in param_list:
    model = keras.Sequential([
        keras.layers.Embedding(len(tokenizer.word_index) + 1, embedding_dim, weights=[embedding_matrix], input_length=max_len, trainable=False),
        keras.layers.Conv1D(filters=params['filters'], kernel_size=3, activation='relu', padding='same'),
        keras.layers.MaxPooling1D(pool_size=2),
        keras.layers.GlobalMaxPooling1D(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(params['dropout_rate']),
        keras.layers.Dense(5, activation='softmax')
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=params['learning_rate']),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(X_train_pad, y_train_emb, epochs=10, batch_size=32, validation_data=(X_val_pad, y_val_emb), verbose=0)
    y_pred = np.argmax(model.predict(X_test_pad), axis=1)
    results_tuning.append({
        'filters': params['filters'],
        'dropout_rate': params['dropout_rate'],
        'learning_rate': params['learning_rate'],
        'accuracy': accuracy_score(y_test_emb, y_pred),
        'precision': precision_score(y_test_emb, y_pred, average='weighted'),
        'recall': recall_score(y_test_emb, y_pred, average='weighted'),
        'f1': f1_score(y_test_emb, y_pred, average='weighted')
    })
print("\nCNN Hyperparameter Tuning Results:")
print(pd.DataFrame(results_tuning))

# %%
nn_configurations = [
    {'hidden_layer_sizes': [512, 256], 'activation': 'relu', 'optimizer': 'adam', 'learning_rate': 0.01},
    {'hidden_layer_sizes': [256, 128], 'activation': 'relu', 'optimizer': 'adam', 'learning_rate': 0.01},
    {'hidden_layer_sizes': [128], 'activation': 'tanh', 'optimizer': 'adam', 'learning_rate': 0.01},
    {'hidden_layer_sizes': [512], 'activation': 'relu', 'optimizer': 'sgd', 'learning_rate': 0.01},
]

results = []
for config in nn_configurations:
    result = train_and_evaluate_nn(**config, num_epochs=10)
    results.append(result)

df_results = pd.DataFrame(results)
print("\nHyperparameter Tuning Results:")
print(df_results)

# %% [markdown]
# ### Hyperparameter Tuning Summary
# 
# We performed an initial round of hyperparameter tuning for the Neural Network model, exploring different combinations of:
# 
# - Hidden Layer Sizes (depth and width of the network)
# - Activation Functions (ReLU vs Tanh)
# - Optimizers (Adam vs SGD)
# - Learning Rate (fixed at 0.01 for this round)
# 
# <br>
# 
# #### Summary of Results
# 
# | Configuration | Train Accuracy | Validation Accuracy | F1 Score | Notes |
# |---|---|---|---|---|
# | [512, 256], relu, adam | 0.7009 | 0.6381 | 0.6068 | Strong balanced performance |
# | [256, 128], relu, adam | 0.7029 | 0.6428 | 0.6146 | Smaller network performs just as well |
# | [128], tanh, adam | 0.6641 | 0.6268 | 0.6118 | Overfitting (train much higher than val) |
# | [512], relu, sgd | 0.5185 | 0.5264 | 0.3917 | Poor training and generalization (SGD struggles) |
# 
# <br>
# 
# #### Key Observations
# 
# 1. Adam significantly outperforms SGD.
# 2. Smaller networks (e.g., [256, 128]) achieve nearly the same performance as larger networks, suggesting that the task does not require extremely deep models.
# 3. Tanh tends to overfit, reaching very high training accuracy but much lower validation accuracy.
# 4. ReLU with Adam emerges as the most consistent and stable choice, with solid performance across different layer sizes.
# 
# <br>
# 
# #### Conclusion
# 
# Based on these results, the recommended configuration for further refinement is:
# 
# - Hidden Layer Sizes: [256, 128]
# - Activation: ReLU
# - Optimizer: Adam
# - Learning Rate: 0.01
# 
# This architecture achieves a good balance between performance and simplicity, making it a strong candidate for further tuning (e.g., adjusting learning rate, adding regularization, increasing epoch count, and exploring batch normalization).
# 

# %% [markdown]
# ---
# ### Step 9: Next Step
# 1. **Fine-tune Neural Network (MLP)** – Furthur adjust hidden layers, learning rate, dropout, batch size, and optimizer to improve performance.
# 2. **Consider LSTM or BERT** – Explore deep NLP models that can capture   contextual information more effectively.
# 3. **Error Analysis** – Review misclassified samples to refine preprocessing and feature engineering.

# %%



