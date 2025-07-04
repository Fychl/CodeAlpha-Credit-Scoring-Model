import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, classification_report

def simulate_data(n_samples=1000, random_state=42):
    np.random.seed(random_state)
    # Simulate features
    income = np.random.normal(50000, 15000, n_samples)  # Annual income
    debts = np.random.normal(15000, 7000, n_samples)    # Total debts
    payment_history = np.random.binomial(1, 0.8, n_samples)  # 1 if good payment history, else 0
    age = np.random.randint(18, 70, n_samples)          # Age of individual
    credit_utilization = np.random.uniform(0, 1, n_samples)  # Credit utilization ratio

    # Simulate target variable: creditworthy (1) or not (0)
    # Simple rule-based generation for demonstration
    creditworthy = ((income > 40000) & (debts < 20000) & (payment_history == 1) & (credit_utilization < 0.5)).astype(int)

    data = pd.DataFrame({
        'income': income,
        'debts': debts,
        'payment_history': payment_history,
        'age': age,
        'credit_utilization': credit_utilization,
        'creditworthy': creditworthy
    })
    return data

def feature_engineering(df):
    # Example feature engineering: debt to income ratio
    df['debt_to_income_ratio'] = df['debts'] / df['income']
    # Drop original debts and income to avoid multicollinearity
    df = df.drop(columns=['debts', 'income'])
    return df

def train_and_evaluate(X_train, X_test, y_train, y_test):
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42)
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)

        results[name] = {
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'ROC-AUC': roc_auc
        }

    return results

def main():
    # Simulate data
    data = simulate_data()

    # Feature engineering
    data = feature_engineering(data)

    # Split data
    X = data.drop(columns=['creditworthy'])
    y = data['creditworthy']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train and evaluate models
    results = train_and_evaluate(X_train, X_test, y_train, y_test)

    # Print results
    for model_name, metrics in results.items():
        print(f"Model: {model_name}")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")
        print()

if __name__ == "__main__":
    main()
