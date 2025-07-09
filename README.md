# CodeAlpha-Credit-Scoring-Model

📊 Credit Scoring Model
This project implements a basic credit scoring model as part of an educational or prototyping task. It demonstrates how to:

Simulate synthetic credit data

Perform feature engineering

Train and evaluate multiple classification models for creditworthiness prediction

✅ Features
Simulates synthetic dataset including:

Annual income

Total debts

Payment history

Age

Credit utilization

Creates a new feature: debt-to-income ratio

Compares three machine learning models:

Logistic Regression

Decision Tree

Random Forest

Evaluates models using:

Precision

Recall

F1-Score

ROC-AUC

🛠 Installation
bash
Copy
Edit
pip install numpy pandas scikit-learn
🚀 Usage
Run the script directly:

bash
Copy
Edit
python credit_scoring.py
The script will:

Simulate data

Engineer features

Train models

Print evaluation metrics

📂 Code Structure
File	Description
credit_scoring.py	Main script with data simulation, training, and evaluation logic

📈 Model Evaluation
After running, you will see printed evaluation metrics for each model, e.g.:

yaml
Copy
Edit
Model: Logistic Regression
  Precision: 0.82
  Recall: 0.78
  F1-Score: 0.80
  ROC-AUC: 0.85
...
✏️ Author
This project was created as part of a CodeAlpha internship Task 1.

