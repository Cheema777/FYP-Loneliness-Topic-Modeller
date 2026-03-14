import pandas as pd
import joblib
from bertopic import BERTopic
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from warnings import filterwarnings
import sys
import os
from dotenv import load_dotenv
from openai import OpenAI

# Suppress warnings for cleaner output
filterwarnings('ignore')
load_dotenv()

# OpenAI setup
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
GPT_MODEL = "gpt-5-mini-2025-08-07"

print("Loading test dataset...")
test_df = pd.read_csv('data/test_dataset.csv')

# Check if True_Theme is filled out
if test_df['True_Theme'].isnull().all() or (test_df['True_Theme'] == "").all():
    print("\nERROR: The 'True_Theme' column in 'data/test_dataset.csv' is empty.")
    print("Please manually fill in the true themes for the 200 rows before running this script.")
    print("Valid themes are:")
    print(" - Emotional Distress")
    print(" - Situational Factors")
    print(" - Existential Loneliness")
    print(" - Social Connection")
    print(" - Communication Barriers")
    sys.exit(1)

# Drop any rows where True_Theme might still be empty to avoid math errors
test_df = test_df.dropna(subset=['True_Theme'])
texts = test_df['Text'].tolist()
true_labels = test_df['True_Theme'].tolist()

THEME_LABELS = {
    0: "Emotional Distress",
    1: "Situational Factors",
    2: "Existential Loneliness",
    3: "Social Connection",
    4: "Communication Barriers"
}
VALID_THEMES = list(THEME_LABELS.values())

print("Loading Models...")
vectorizer = joblib.load('models/vectorizer.pkl')
nmf_engine = joblib.load('models/nmf_model.pkl')
lda_engine = joblib.load('models/lda_model.pkl')
bert_engine = BERTopic.load("models/loneliness_bertopic_model")

# Dictionary to hold predictions
predictions = {
    'NMF': [],
    'LDA': [],
    'BERTopic': [],
    'GPT': []
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
found_topics, probabilities = bert_engine.transform(texts)
for i in range(len(texts)):
    if found_topics[i] != -1:
        assigned_id = int(found_topics[i])
    else:
        assigned_id = int(probabilities[i].argmax())
    predictions['BERTopic'].append(THEME_LABELS.get(assigned_id, "Unclassified"))

GPT_SYSTEM_PROMPT = """You are an expert loneliness theme classifier trained on a research dataset about what loneliness means to people. You must classify text into exactly ONE of these 5 loneliness themes:

1. SOCIAL CONNECTION — The person describes lacking people, companionship, friends, family, or someone to spend time with. This is the MOST COMMON theme. Look for: "no one to talk to", "having no friends", "no one around", "being on your own", "not having people", "nobody to share with".
   Examples:
   - "Feeling like you have no one to talk to or spend time with or to support you when you are in need."
   - "When you feel like s...! There is nobody to make you a coffee or around of toast."
   - "Loneliness means not having another voice/presence there after having them there."

2. COMMUNICATION BARRIERS — The person describes inability to express feelings, not being understood, or lacking mutual understanding. Look for: "no one understands", "can't express", "not being heard", "feeling misunderstood".
   Examples:
   - "Having no one to share experiences and dreams with. Feeling like no one understands or values me."
   - "Loneliness is the experience of not being able to say what one really thinks or feel, because it would be unacceptable."

3. EXISTENTIAL LONELINESS — The person describes deep disconnection, alienation, or philosophical emptiness. Look for: "disconnected", "alienation", "empty inside", "cut off from the world".
   Examples:
   - "Alienation, no one to share activities with, no one to chat with."
   - "Feeling disconnected from others."

4. SITUATIONAL FACTORS — The person describes specific life circumstances causing loneliness such as living alone, weekends/holidays, bereavement, illness, retirement. Look for: "empty house", "weekends", "Christmas", "after partner died", "moved to new area".
   Examples:
   - "Going home to a empty house. Doing things on your own."
   - "Not having people whom I can be with at weekends easter christmas or other non-work days."

5. EMOTIONAL DISTRESS — The person primarily describes raw emotions: sadness, pain, depression, despair WITHOUT social or situational context. ONLY choose this when the text is purely about emotions.
   Examples:
   - "Sadness. Detachment."
   - "Sadness. Irritating. Tired."

IMPORTANT RULES:
- Social Connection is the most common theme — if someone describes lacking people or companionship, choose Social Connection, NOT Emotional Distress.
- Choose Emotional Distress ONLY when the text is purely about emotions without describing social, situational, or communication aspects.
- If the text mentions specific life situations (house, work, holidays), choose Situational Factors.
- If the text emphasises not being understood or inability to express, choose Communication Barriers.
- If the text describes deep disconnection or alienation, choose Existential Loneliness."""

print("Running GPT Predictions... (this may take several minutes)")
for i, text in enumerate(texts):
    try:
        user_prompt = f"""Classify this text into exactly ONE loneliness theme. Reply with ONLY the theme name, nothing else.

Text: "{text}"
"""
        response = openai_client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": GPT_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            max_completion_tokens=1000
        )
        theme = response.choices[0].message.content
        if not theme:
            predictions['GPT'].append("Unclassified")
            continue
        theme = theme.strip()
        # Match to valid theme
        matched = "Unclassified"
        for valid in VALID_THEMES:
            if valid.lower() in theme.lower():
                matched = valid
                break
        predictions['GPT'].append(matched)
    except Exception as e:
        print(f"  GPT Error on row {i}: {e}")
        predictions['GPT'].append("Unclassified")
    
    if (i + 1) % 20 == 0:
        print(f"  GPT: {i+1}/{len(texts)} done...")

print("\n" + "="*50)
print("             MODEL EVALUATION REPORT              ")
print("="*50)

for model_name, preds in predictions.items():
    acc = accuracy_score(true_labels, preds)
    # Using 'weighted' since themes might be imbalanced
    prec = precision_score(true_labels, preds, average='weighted', zero_division=0)
    rec = recall_score(true_labels, preds, average='weighted', zero_division=0)
    f1 = f1_score(true_labels, preds, average='weighted', zero_division=0)
    
    print(f"\n{model_name} Model:")
    print(f"  Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Precision: {prec:.4f} ({prec*100:.2f}%)")
    print(f"  Recall:    {rec:.4f} ({rec*100:.2f}%)")
    print(f"  F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
print("="*50)
