# Resume Screening Web Application - Day 4 Flask Deployment

## Project Overview
Flask-based web application for automated resume screening and job category classification. This is Day 4 implementation of the Resume Screening System, deployed using the ML model trained in Day 3.

## Objective
To build a user-friendly web interface where users can upload resumes in PDF/DOCX/TXT format and receive instant job category predictions with confidence scores and skill matching.

## Key Features
1. **Resume Upload**: Supports PDF, DOCX, and TXT formats up to 2MB
2. **AI Prediction**: Predicts job category from 6 domains using Random Forest model
3. **Confidence Score**: Shows prediction confidence percentage
4. **Skill Extraction**: Displays top matching skills found in resume
5. **Resume Preview**: Shows extracted text summary
6. **Pre-trained Model**: Auto-trains on built-in dataset, no external files required

## Tech Stack
- **Backend**: Flask, Python 3.10+
- **ML Model**: Scikit-learn Random Forest Classifier + TF-IDF Vectorizer
- **Frontend**: HTML5, CSS3
- **File Processing**: PyPDF2 for PDF, python-docx for DOCX

## Dataset
Pre-prepared synthetic dataset with 300 resume samples across 6 categories:
- Data Science
- Software Engineer
- Web Developer  
- HR
- Sales
- Digital Marketing
*No external CSV file required - dataset auto-generates on app startup*

## Installation & Setup

### Prerequisites
```bash
Python 3.8 or higher
pip package manager
### Steps
1. Navigate to project folder:
cd Resume_Screener_Day4
2. Install dependencies:
pip install flask scikit-learn pandas numpy PyPDF2 python-docx joblib
3. Run the application:
python app.py
4. Open browser and visit:
http://127.0.0.1:5000
## Project Structure
Resume_Screener_Day4/
├── app.py                      # Main Flask application
├── templates/
│   ├── index.html             # Upload page
│   └── result.html            # Prediction result page
├── uploads/                   # Uploaded resumes storage
└── README.md                  # This file
## Usage
1. Launch the application using `python app.py`
2. Upload a resume file using the web interface
3. View predicted job category, confidence score, and extracted skills
4. Upload another resume by clicking "Upload Another Resume"

## Model Details
- *Algorithm*: Random Forest Classifier, 100 estimators
- *Vectorization*: TF-IDF with 500 max features
- *Training*: Auto-trains on startup using synthetic dataset
- *Accuracy*: ∼96% on synthetic test data

## Author
Warda Ejaz 
Softgrid AI/ML Intern
Date: 25 June 2026
