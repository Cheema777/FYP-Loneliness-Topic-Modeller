from flask import Flask, render_template, request, redirect
import joblib
from bertopic import BERTopic
import sqlite3
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os

app = Flask(__name__)

# Basic database setup for the history log
def init_db():
    conn = sqlite3.connect('history.db')
    c = conn.cursor()
    # Table structure for tracking user inputs and AI interpretations
    c.execute('''CREATE TABLE IF NOT EXISTS analysis_history
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT, user_text TEXT, 
                  model_used TEXT, detected_theme TEXT)''')
    conn.commit()
    conn.close()

init_db() 

# mapping 5 Themes from my research findings
THEME_LABELS = {
    0: "Emotional Distress",
    1: "Situational Factors",
    2: "Existential Loneliness",
    3: "Social Connection",
    4: "Communication Barriers"
}

# Load saved models from the /models folder
try:
    vectorizer = joblib.load('models/vectorizer.pkl')
    nmf_engine = joblib.load('models/nmf_model.pkl')
    lda_engine = joblib.load('models/lda_model.pkl')
    bert_engine = BERTopic.load("models/loneliness_bertopic_model")
    print("All models loaded.")
except Exception as e:
    print(f"Error loading models: {e}")

# Converting raw scores to percentages for Chart.js
def calculate_probabilities(scores):
    exp_s = np.exp(scores - np.max(scores))
    return (exp_s / exp_s.sum()).tolist()

@app.route('/')
def index():
    # Fetching history to show in the side panel
    conn = sqlite3.connect('history.db')
    db_cursor = conn.cursor()
    db_cursor.execute("SELECT * FROM analysis_history ORDER BY id DESC")
    history_data = db_cursor.fetchall()
    conn.close()
    return render_template('index.html', history=history_data)

@app.route('/predict', methods=['POST'])
def run_analysis():
    input_text = request.form['message']
    selected_model = request.form['model_choice']
    # Defaults
    assigned_id = 0
    strength_scores = []
    #Logic for  (NMF & LDA)
    if selected_model in ['NMF', 'LDA']:
        current_engine = nmf_engine if selected_model == 'NMF' else lda_engine
        # vectorizing input using the same vocabulary the models were trained on
        text_vector = vectorizer.transform([input_text])
        raw_outputs = current_engine.transform(text_vector)[0]
        
        assigned_id = int(raw_outputs.argmax())
        strength_scores = [round(x * 100, 2) for x in calculate_probabilities(raw_outputs)]
    #Logic for Transformer Model
    elif selected_model == 'BERTopic':
        found_topics, probabilities = bert_engine.transform([input_text])
        
        # BERTopic returns -1 if it's "noise." I've added a fallback to the max probability 
        # so the user always gets a thematic result.
        if found_topics[0] != -1:
            assigned_id = int(found_topics[0])
        else:
            assigned_id = int(probabilities[0].argmax())
        #extracting the scores for the 5 themes    
        strength_scores = [round(float(x) * 100, 2) for x in probabilities[0][:5]]
    # Map the numeric ID to the final string theme
    final_theme = THEME_LABELS.get(assigned_id, "Unclassified")

    # Logging to history.db
    conn = sqlite3.connect('history.db')
    db_cursor = conn.cursor()
    db_cursor.execute("INSERT INTO analysis_history (timestamp, user_text, model_used, detected_theme) VALUES (?, ?, ?, ?)",
                      (datetime.now().strftime("%H:%M - %d %b"), input_text, selected_model, final_theme))
    conn.commit()
    # reload so the new result appear immediately
    db_cursor.execute("SELECT * FROM analysis_history ORDER BY id DESC")
    updated_history = db_cursor.fetchall()
    conn.close()

    return render_template('index.html', 
                           prediction=final_theme, 
                           original_text=input_text, 
                           model_used=selected_model,
                           history=updated_history,
                           chart_data=strength_scores)

@app.route('/clear', methods=['POST'])
def reset_history():
    #function to wipe the database for a fresh test run
    conn = sqlite3.connect('history.db')
    db_cursor = conn.cursor()
    db_cursor.execute("DELETE FROM analysis_history")
    conn.commit()
    conn.close()
    return redirect('/')

@app.route('/compare')
def compare_models():
    # Load test dataset
    test_df = pd.read_csv('data/test_dataset.csv')
    test_df = test_df.dropna(subset=['True_Theme'])
    texts = test_df['Text'].tolist()
    true_labels = test_df['True_Theme'].tolist()

    theme_names = list(THEME_LABELS.values())

    # --- Run all 3 models ---
    predictions = {}

    # NMF
    text_vectors = vectorizer.transform(texts)
    nmf_raw = nmf_engine.transform(text_vectors)
    predictions['NMF'] = [THEME_LABELS.get(int(row.argmax()), "Unclassified") for row in nmf_raw]

    # LDA
    lda_raw = lda_engine.transform(text_vectors)
    predictions['LDA'] = [THEME_LABELS.get(int(row.argmax()), "Unclassified") for row in lda_raw]

    # BERTopic
    found_topics, probabilities = bert_engine.transform(texts)
    bert_preds = []
    for i in range(len(texts)):
        if found_topics[i] != -1:
            assigned_id = int(found_topics[i])
        else:
            assigned_id = int(probabilities[i].argmax())
        bert_preds.append(THEME_LABELS.get(assigned_id, "Unclassified"))
    predictions['BERTopic'] = bert_preds

    # --- Compute metrics ---
    metrics = {}
    conf_matrices = {}
    for model_name, preds in predictions.items():
        acc = accuracy_score(true_labels, preds)
        prec = precision_score(true_labels, preds, average='weighted', zero_division=0)
        rec = recall_score(true_labels, preds, average='weighted', zero_division=0)
        f1 = f1_score(true_labels, preds, average='weighted', zero_division=0)
        metrics[model_name] = {
            'accuracy': round(acc * 100, 2),
            'precision': round(prec * 100, 2),
            'recall': round(rec * 100, 2),
            'f1': round(f1 * 100, 2)
        }
        # Confusion matrix (rows = true, cols = predicted)
        cm = confusion_matrix(true_labels, preds, labels=theme_names)
        conf_matrices[model_name] = cm.tolist()

    return render_template('compare.html',
                           metrics=metrics,
                           conf_matrices=conf_matrices,
                           theme_names=theme_names)


if __name__ == '__main__':
    app.run(debug=True)