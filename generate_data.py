import pandas as pd
import joblib
from bertopic import BERTopic
import numpy as np

# Load original dataset
df = pd.read_excel('data/Whatisloneliness060321.xlsx')

# Filter out empty or numeric rows (like -99)
df = df[df['Q9_Loneliness_Meaning_Qual'].apply(lambda x: isinstance(x, str) and len(x) > 10)]

# Take a random sample of 200 responses
test_df = df.sample(n=200, random_state=42)[['Idme', 'Q9_Loneliness_Meaning_Qual']].copy()
test_df.columns = ['ID', 'Text']

print("Loading BERTopic for ground truth generation...")
bert_engine = BERTopic.load("models/loneliness_bertopic_model")

THEME_LABELS = {
    0: "Emotional Distress",
    1: "Situational Factors",
    2: "Existential Loneliness",
    3: "Social Connection",
    4: "Communication Barriers"
}

print("Predicting ground truths...")
texts = test_df['Text'].tolist()
found_topics, probabilities = bert_engine.transform(texts)

true_themes = []
for i in range(len(texts)):
    if found_topics[i] != -1:
        assigned_id = int(found_topics[i])
    else:
        assigned_id = int(probabilities[i].argmax())
        
    final_theme = THEME_LABELS.get(assigned_id, "Unclassified")
    true_themes.append(final_theme)

test_df['True_Theme'] = true_themes

# Save to CSV
test_df.to_csv('data/test_dataset.csv', index=False)
print("Saved 200 rows to data/test_dataset.csv!")
