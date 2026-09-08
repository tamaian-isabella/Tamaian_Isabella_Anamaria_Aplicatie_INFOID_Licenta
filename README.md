# Sistem Software pentru Estimarea Riscului de Boli Cardiovasculare

Acest repository conține întregul cod sursă al lucrării de licență destinate evaluării riscului cardiovascular pe baza parametrilor clinici, utilizând algoritmi de clasificare și analiză explicativă prin SHAP.

---

## 1. Structura Repository-ului

- `train_model.py` — Procesul de antrenare, scalare Z-score și benchmarking pe cei 8 algoritmi Scikit-Learn.
- `app.py` — Aplicația web interactivă dezvoltată în Streamlit
- `data/` — Directorul ce conține setul de date clinic de referință (`processed.cleveland.data`).
- `requirements.txt` — Lista completă a dependențelor Python necesare execuției.
- `.gitignore` — Exclude fișierele binare, fișierele cache și mediile virtuale.

---

## 2. Cerințe de Sistem

- Python 3.9+
- Git

---

## 3. Ghid de Instalare, Compilare și Lansare

Urmați pașii de mai jos în terminal pentru configurarea completă a proiectului pe un sistem nou:

### 3.1. Pe sisteme Windows (PowerShell):

```powershell
# Pasul 1: Clonarea repository-ului
git clone [https://github.com/tamaian-isabella/Tamaian_Isabella_Anamaria_Aplicatie_INFOID_Licenta.git](https://github.com/tamaian-isabella/Tamaian_Isabella_Anamaria_Aplicatie_INFOID_Licenta.git)
cd Tamaian_Isabella_Anamaria_Aplicatie_INFOID_Licenta

# Pasul 2: Crearea și activarea mediului virtual
python -m venv heart_env
.\heart_env\Scripts\Activate.ps1

# Pasul 3: Instalarea dependențelor
pip install --upgrade pip
pip install -r requirements.txt

# Pasul 4: Generarea artefactelor și antrenarea modelelor
python train_model.py

# Pasul 5: Lansarea aplicației interactive
streamlit run app.py
```

### 3.2. Pe sisteme Linux / macOS (sau Git Bash pe Windows):

```bash
# Pasul 1: Clonarea repository-ului
git clone [https://github.com/tamaian-isabella/Tamaian_Isabella_Anamaria_Aplicatie_INFOID_Licenta.git](https://github.com/tamaian-isabella/Tamaian_Isabella_Anamaria_Aplicatie_INFOID_Licenta.git)
cd Tamaian_Isabella_Anamaria_Aplicatie_INFOID_Licenta

# Pasul 2: Crearea și activarea mediului virtual
python3 -m venv heart_env
source heart_env/bin/activate

# Pasul 3: Instalarea dependențelor
pip install --upgrade pip
pip install -r requirements.txt

# Pasul 4: Generarea artefactelor și antrenarea modelelor
python train_model.py

# Pasul 5: Lansarea aplicației interactive
streamlit run app.py
```

Aplicația va fi accesibilă în browser la adresa: `http://localhost:8501`.
