import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Algoritmi
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Incarcarea datelor
column_names = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
    'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 
    'ca', 'thal', 'target'
]

df = pd.read_csv(
    "data/processed.cleveland.data", 
    names=column_names, 
    na_values='?'
)

# Curatarea randurilor cu valori lipsa
df = df.dropna()

# Conversia tintei in format binar: 0 = Fara boala, 1 = Prezinta boala
df['target'] = (df['target'] > 0).astype(int)

# 2. Separarea caracteristicilor (X) si a variabilei tintă (y)
X = df.drop(columns=['target'])
y = df['target']

# 3. Impartirea in seturi de Antrenare si Testare (80% - 20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Standardizarea caracteristicilor

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Conversia inapoi in DataFrame pentru pastrarea numelor coloanelor (necesar pentru SHAP)
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X.columns)

# 5. Definirea listei de algoritmi in limba engleza
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Support Vector Machine": SVC(probability=True, random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Naive Bayes": GaussianNB(),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "MLP Neural Network": MLPClassifier(max_iter=1000, random_state=42)
}

# 6. Evaluarea si compararea modelelor
results = []
best_model = None
best_model_name = ""
best_acc = 0.0

for name, model in models.items():
    model.fit(X_train_scaled_df, y_train)
    preds = model.predict(X_test_scaled_df)
    
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    
    results.append({
        "Algorithm": name,
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1 Score": round(f1, 4)
    })
    
    if acc > best_acc:
        best_acc = acc
        best_model = model
        best_model_name = name

print(f"Cel mai bun model: {best_model_name} cu o Acuratețe de: {best_acc:.4f}")

# Salvarea metricilor comparative intr-un fisier CSV
metrics_df = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False)
metrics_df.to_csv("model_metrics.csv", index=False)

# Salvarea celui mai bun model, a scaler-ului si a datelor de antrenament pentru SHAP
joblib.dump(best_model, "best_heart_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(X_train_scaled_df, "X_train_scaled.pkl")

print("Antrenare finalizată. Modelele au fost evaluate, iar fișierele au fost salvate cu succes.")