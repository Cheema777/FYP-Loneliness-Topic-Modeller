import pandas as pd
import joblib
from bertopic import BERTopic
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, f1_score
from warnings import filterwarnings

# Suppress warnings for cleaner output
filterwarnings('ignore')

print("Loading test dataset...")
test_df = pd.read_csv('data/test_dataset.csv')
texts = test_df['Text'].tolist()
true_labels = test_df['True_Theme'].tolist()

THEME_LABELS = {
    0: "Emotional Distress",
    1: "Situational Factors",
    2: "Existential Loneliness",
    3: "Social Connection",
    4: "Communication Barriers"
}

print("Loading Models...")
vectorizer = joblib.load('models/vectorizer.pkl')
nmf_engine = joblib.load('models/nmf_model.pkl')
lda_engine = joblib.load('models/lda_model.pkl')
bert_engine = BERTopic.load("models/loneliness_bertopic_model")

# Dictionary to hold predictions
predictions = {
    'NMF': [],
    'LDA': [],
    'BERTopic': []
}

print("Running NMF Predictions...")
text_vectors = vectorizer.transform(texts)
nmf_raw = nmf_engine.transform(text_vectors)
for row in nmf_raw:
    assigned_id = int(row.argmax())
    predictions['NMF'].append(THEME_LABELS.get(assigned_id, "Unclassified"))

print("Running LDA Predictions...")
lda_raw = lda_engine.transform(text_vectors)
for row in lda_raw:
    assigned_id = int(row.argmax())
    predictions['LDA'].append(THEME_LABELS.get(assigned_id, "Unclassified"))

print("Running BERTopic Predictions...")
# Since we used BERTopic as ground truth, it should score 100%, 
# but we run it anyway for completeness of script logic.
found_topics, probabilities = bert_engine.transform(texts)
for i in range(len(texts)):
    if found_topics[i] != -1:
        assigned_id = int(found_topics[i])
    else:
        assigned_id = int(probabilities[i].argmax())
    predictions['BERTopic'].append(THEME_LABELS.get(assigned_id, "Unclassified"))

print("\n" + "="*50)
print("             MODEL EVALUATION REPORT              ")
print("="*50)

for model_name, preds in predictions.items():
    acc = accuracy_score(true_labels, preds)
    # Using 'weighted' since themes might be imbalanced
    prec = precision_score(true_labels, preds, average='weighted')
    f1 = f1_score(true_labels, preds, average='weighted')
    
    print(f"\n{model_name} Model:")
    print(f"  Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Precision: {prec:.4f} ({prec*100:.2f}%)")
    print(f"  F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
print("="*50)

