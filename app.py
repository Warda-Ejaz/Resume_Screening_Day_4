from flask import Flask, render_template, request, redirect, url_for
import joblib
import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import PyPDF2
import docx
import re

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 # 2MB limit

# Global variables
model = None
vectorizer = None

# Pre-prepared dataset generator - NO CSV NEEDED
def create_and_train_model():
    print("Creating fake dataset and training model...")

    categories = ['Data Science', 'Software Engineer', 'Web Developer', 'HR', 'Sales', 'Digital Marketing']
    resume_samples = {
        'Data Science': [
            'Python developer with ML, Pandas, Numpy, TensorFlow, SQL experience',
            'Data Scientist skilled in Machine Learning, Statistics, Python, R, Tableau',
            'Expert in Python, Scikit-learn, Data Visualization, SQL, AI projects'
        ],
        'Software Engineer': [
            'Java developer with Spring Boot, Microservices, REST API, MySQL experience',
            'Software Engineer C++ Java Python. DSA, System Design, Git, Docker',
            'Full Stack Java developer. Spring, Hibernate, AWS, Jenkins'
        ],
        'Web Developer': [
            'Frontend developer React JavaScript HTML CSS. Responsive design Bootstrap',
            'Web developer MERN stack. MongoDB Express React Node.js e-commerce',
            'UI/UX developer HTML5 CSS3 JavaScript. WordPress PHP SEO'
        ],
        'HR': [
            'HR Manager recruitment employee relations payroll performance management',
            'Human Resource talent acquisition onboarding HR policies labor laws',
            'HR Executive interviewing training employee engagement compliance'
        ],
        'Sales': [
            'Sales Executive B2B sales client relationship target achievement CRM',
            'Sales Manager lead generation negotiation market research field sales',
            'Retail sales customer service product knowledge sales targets'
        ],
        'Digital Marketing': [
            'Digital Marketer SEO SEM Google Ads Facebook Ads content marketing',
            'Social Media Manager Instagram Facebook LinkedIn campaigns analytics',
            'Marketing Executive email marketing PPC Google Analytics brand management'
        ]
    }

    data = []
    for cat, resumes in resume_samples.items():
        for _ in range(50):
            resume = np.random.choice(resumes)
            data.append({'Category': cat, 'Resume_Text': resume})

    df = pd.DataFrame(data)
    df['Resume_Text'] = df['Resume_Text'].str.lower().str.replace(r'[^a-zA-Z\s]', '', regex=True)

    # Train model
    vec = TfidfVectorizer(max_features=500, stop_words='english')
    X = vec.fit_transform(df['Resume_Text'])
    y = df['Category']

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)

    print("Model trained successfully!")
    return clf, vec

def extract_text_from_file(filepath):
    text = ""
    if filepath.endswith('.pdf'):
        with open(filepath, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text()
    elif filepath.endswith('.docx'):
        doc = docx.Document(filepath)
        for para in doc.paragraphs:
            text += para.text + "\n"
    else: # txt
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
    return text

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'resume' not in request.files:
            return redirect(request.url)

        file = request.files['resume']
        if file.filename == '':
            return redirect(request.url)

        if file and file.filename.endswith(('.pdf', '.docx', '.txt')):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            # Extract and predict
            resume_text = extract_text_from_file(filepath)
            clean_resume = clean_text(resume_text)
            features = vectorizer.transform([clean_resume])
            prediction = model.predict(features)[0]
            probabilities = model.predict_proba(features)[0]
            confidence = max(probabilities) * 100

            # Matching skills - simple extraction
            skills = ['Python', 'Java', 'React', 'SQL', 'ML', 'AWS', 'HTML', 'CSS', 'SEO', 'HR']
            found_skills = [skill for skill in skills if skill.lower() in clean_resume]

            return render_template('result.html',
                                 category=prediction,
                                 confidence=round(confidence, 2),
                                 skills=found_skills[:5],
                                 resume_preview=resume_text[:300])

    return render_template('index.html')
if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    
    # Model ko 1 dafa hi train karo, har reload pe nahi
    model, vectorizer = create_and_train_model()
    
    app.run(debug=False, port=5000) # debug=False kar diya


