from flask import Flask, render_template, request, redirect
import joblib
from bertopic import BERTopic
import sqlite3
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

app = Flask(__name__)

# OpenAI client setup
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
GPT_MODEL = "gpt-5-mini-2025-08-07"

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

VALID_THEMES = list(THEME_LABELS.values())

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

# Shared system prompt for GPT — detailed theme definitions with examples from the dataset
GPT_SYSTEM_PROMPT = """You are an expert loneliness theme classifier trained on a research dataset about what loneliness means to people. You must classify text into exactly ONE of these 5 loneliness themes:

1. SOCIAL CONNECTION — The person describes lacking people, companionship, friends, family, or someone to spend time with. This is the MOST COMMON theme. Look for: "no one to talk to", "having no friends", "no one around", "being on your own", "not having people", "nobody to share with".
   Examples:
   - "Feeling like you have no one to talk to or spend time with or to support you when you are in need."
   - "When you feel like s...! There is nobody to make you a coffee or around of toast."
   - "Loneliness means not having another voice/presence there after having them there."
   - "Not having people around you or available to talk to."

2. COMMUNICATION BARRIERS — The person describes inability to express feelings, not being understood, or lacking mutual understanding and trust. Look for: "no one understands", "can't express", "not being heard", "feeling misunderstood", "unacceptable to say what you think".
   Examples:
   - "Having no one to share experiences and dreams with. Feeling like no one understands or values me."
   - "Loneliness is the experience of not being able to say what one really thinks or feel, because it would be unacceptable."
   - "An unfulfilled longing for companionship where there is mutual trust and a click of common understanding."

3. EXISTENTIAL LONELINESS — The person describes deep disconnection, alienation, or philosophical emptiness. Look for: "disconnected", "alienation", "empty inside", "cut off from the world", "meaningless".
   Examples:
   - "Alienation, no one to share activities with, no one to chat with."
   - "Feeling disconnected from others."
   - "Feeling disconnected."

4. SITUATIONAL FACTORS — The person describes specific life circumstances causing loneliness such as living alone, weekends/holidays, bereavement, illness, retirement, or relocation. Look for: "empty house", "weekends", "Christmas", "after partner died", "moved to new area", "retired".
   Examples:
   - "Going home to a empty house. Doing things on your own. Having no one to talk to when you have a bad day at work."
   - "Not having people whom I can be with at weekends easter christmas or other non-work days."
   - "A mental state. Physically not being near someone, especially at home."

5. EMOTIONAL DISTRESS — The person primarily describes raw emotions: sadness, pain, depression, despair, or suffering WITHOUT specific situational or social context. This should ONLY be chosen when the text is purely about emotions. Look for: "sadness", "pain", "depressed", "despair", "hurts", "horrible feeling".
   Examples:
   - "Sadness. Detachment."
   - "Sadness. Irritating. Tired."
   - "A dreadful, horrible painful experience."

IMPORTANT RULES:
- Social Connection is the most common theme — if someone describes lacking people or companionship, choose Social Connection, NOT Emotional Distress.
- Choose Emotional Distress ONLY when the text is purely about emotions/feelings without describing social, situational, or communication aspects.
- If the text mentions specific life situations (house, work, holidays), choose Situational Factors.
- If the text emphasises not being understood or inability to express, choose Communication Barriers.
- If the text describes deep disconnection or alienation, choose Existential Loneliness."""

# GPT theme classification using structured prompt
def classify_with_gpt(text):
    """Send text to GPT and get a theme classification with confidence scores."""
    user_prompt = f"""Classify this text into exactly ONE of the 5 loneliness themes.

Text: "{text}"

Respond in EXACTLY this format (no extra text):
THEME: <one of: Emotional Distress, Situational Factors, Existential Loneliness, Social Connection, Communication Barriers>
SCORES: <5 comma-separated confidence percentages that sum to 100, in order: Emotional Distress, Situational Factors, Existential Loneliness, Social Connection, Communication Barriers>"""

    try:
        response = openai_client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": GPT_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            max_completion_tokens=2000
        )
        reply = response.choices[0].message.content
        if not reply:
            print(f"GPT returned empty content. Finish reason: {response.choices[0].finish_reason}")
            return "Unclassified", [20.0, 20.0, 20.0, 20.0, 20.0]
        reply = reply.strip()
        
        # Parse the response
        lines = reply.split('\n')
        theme = None
        scores = [20.0, 20.0, 20.0, 20.0, 20.0]  # default equal scores
        
        for line in lines:
            if line.upper().startswith('THEME:'):
                theme = line.split(':', 1)[1].strip()
            elif line.upper().startswith('SCORES:'):
                try:
                    score_str = line.split(':', 1)[1].strip()
                    scores = [float(s.strip().replace('%', '')) for s in score_str.split(',')]
                    if len(scores) != 5:
                        scores = [20.0, 20.0, 20.0, 20.0, 20.0]
                except:
                    scores = [20.0, 20.0, 20.0, 20.0, 20.0]
        
        # Validate theme — fuzzy match to valid themes
        if theme not in VALID_THEMES:
            for valid in VALID_THEMES:
                if valid.lower() in theme.lower():
                    theme = valid
                    break
            else:
                theme = VALID_THEMES[scores.index(max(scores))]
        
        return theme, [round(s, 2) for s in scores]
    except Exception as e:
        print(f"GPT Error: {e}")
        return "Unclassified", [20.0, 20.0, 20.0, 20.0, 20.0]

# GPT classification for evaluation (theme name only, no scores needed)
def classify_with_gpt_simple(text):
    """Simplified GPT call for batch evaluation — returns theme name only."""
    user_prompt = f"""Classify this text into exactly ONE loneliness theme. Reply with ONLY the theme name, nothing else.

Text: "{text}"
"""
    try:
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
            return "Unclassified"
        theme = theme.strip()
        # Clean up any extra formatting
        for valid in VALID_THEMES:
            if valid.lower() in theme.lower():
                return valid
        return "Unclassified"
    except Exception as e:
        print(f"GPT Error: {e}")
        return "Unclassified"

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
    final_theme = "Unclassified"

    #Logic for  (NMF & LDA)
    if selected_model in ['NMF', 'LDA']:
        current_engine = nmf_engine if selected_model == 'NMF' else lda_engine
        # vectorizing input using the same vocabulary the models were trained on
        text_vector = vectorizer.transform([input_text])
        raw_outputs = current_engine.transform(text_vector)[0]
        
        assigned_id = int(raw_outputs.argmax())
        strength_scores = [round(x * 100, 2) for x in calculate_probabilities(raw_outputs)]
        final_theme = THEME_LABELS.get(assigned_id, "Unclassified")

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
        final_theme = THEME_LABELS.get(assigned_id, "Unclassified")

    # Logic for GPT Model
    elif selected_model == 'GPT':
        final_theme, strength_scores = classify_with_gpt(input_text)

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
    import json
    import os

    results_path = os.path.join(os.path.dirname(__file__), 'data', 'model_results.json')
    
    # Check if cache exists
    if not os.path.exists(results_path):
        return "Model results not pre-computed yet. Please run evaluate_models.py first.", 500
        
    with open(results_path, 'r') as f:
        results_data = json.load(f)

    metrics = results_data['metrics']
    conf_matrices = results_data['conf_matrices']
    theme_names = results_data['theme_names']

    return render_template('compare.html',
                           metrics=metrics,
                           conf_matrices=conf_matrices,
                           theme_names=theme_names)


if __name__ == '__main__':
    app.run(debug=True)