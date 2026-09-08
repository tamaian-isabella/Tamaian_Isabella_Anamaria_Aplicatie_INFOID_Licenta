import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from datetime import datetime
from fpdf import FPDF

st.set_page_config(page_title="Estimator Risc Boli Cardiovasculare", layout="wide")

st.title("Sistem software pentru estimarea riscului de boli cardiovasculare bazat pe algoritmi de învățare automată")

st.markdown("""
Această aplicație reprezintă un sistem inteligent de asistență pentru decizii clinice, dezvoltat pentru a evalua riscul de boli cardiovasculare pe baza parametrilor fiziologici ai pacientului. Utilizând modele avansate de Machine Learning antrenate pe setul de date clinic Cleveland, aplicația oferă atât predicții în timp real, cât și explicații vizuale detaliate bazate pe analiza valorilor SHAP.
""")

# Incarcarea modelelor si fisierelor salvate
@st.cache_resource
def load_artifacts():
    model = joblib.load("best_heart_model.pkl")
    scaler = joblib.load("scaler.pkl")
    metrics_df = pd.read_csv("model_metrics.csv")
    X_train_scaled = joblib.load("X_train_scaled.pkl")
    return model, scaler, metrics_df, X_train_scaled

try:
    model, scaler, metrics_df, X_train_scaled = load_artifacts()
except Exception as e:
    st.error("Vă rugăm să rulați mai întâi comanda `python train_model.py` pentru a genera fișierele modelului.")
    st.stop()

# Functie pentru curatarea textului pentru FPDF (compatibilitate caracter Latin-1)
def clean_pdf_text(text):
    replacements = {
        'ă': 'a', 'Ă': 'A', 'â': 'a', 'Â': 'A',
        'î': 'i', 'Î': 'I', 'ș': 's', 'Ș': 'S',
        'ț': 't', 'Ț': 'T', 'ş': 's', 'Ş': 'S', 'ţ': 't', 'Ţ': 'T'
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text.encode('latin-1', 'replace').decode('latin-1')

# Functie ajutatoare pentru generarea raportului PDF
def generate_pdf_report(c_vals, risk_label, conf_str, timestamp, metrics_df):
    pdf = FPDF()
    pdf.add_page()
    
    # Antet Titlu
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 10, clean_pdf_text("RAPORT CLINIC DE EVALUARE A PACIENTULUI"), ln=1, align="C")
    pdf.set_font("Helvetica", "I", 9)
    pdf.cell(0, 5, clean_pdf_text(f"Generat la data: {timestamp}"), ln=1, align="C")
    pdf.ln(5)
    
    # Rezultat Evaluare
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, clean_pdf_text(f"Rezultat Evaluare: {risk_label}"), ln=1)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, clean_pdf_text(f"Nivel de Incredere Model: {conf_str}"), ln=1)
    pdf.ln(5)
    
    # Sectiunea 1: Date Clinice Pacient
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, clean_pdf_text("1. Parametri Clinici Introdusi"), ln=1)
    pdf.set_font("Helvetica", "", 9)
    
    metrics = [
        ("Varsta", f"{c_vals['age']} ani"),
        ("Sex", f"{c_vals['sex']}"),
        ("Tip Durere Toracica (CP)", f"{c_vals['cp']}"),
        ("Tensiune Arteriala in Repaus", f"{c_vals['trestbps']} mm Hg"),
        ("Colesterol Seric", f"{c_vals['chol']} mg/dl"),
        ("Glicemie in Jeun > 120 mg/dl", f"{c_vals['fbs']}"),
        ("Rezultat EKG in Repaus", f"Tip {c_vals['restecg']}"),
        ("Frecventa Cardiaca Maxima", f"{c_vals['thalach']} bpm"),
        ("Angina Indusa de Efort", f"{c_vals['exang']}"),
        ("Depresie ST (Oldpeak)", f"{c_vals['oldpeak']}"),
        ("Panta Segmentului ST", f"Tip {c_vals['slope']}"),
        ("Nivel Vase Majore (Fluoroscopie)", f"{c_vals['ca']} vase"),
        ("Rezultat Talasemie", f"Tip {c_vals['thal']}"),
    ]
    
    for key, val in metrics:
        pdf.cell(80, 5, clean_pdf_text(f"  - {key}:"), ln=0)
        pdf.cell(0, 5, clean_pdf_text(str(val)), ln=1)
    
    pdf.ln(5)
    
    # Sectiunea 2: Detalii Model & Pipeline ML
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, clean_pdf_text("2. Detalii Model & Pipeline de Machine Learning"), ln=1)
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(0, 5, clean_pdf_text("  - Biblioteca Software: Scikit-Learn (Python ML Ecosystem)"), ln=1)
    pdf.cell(0, 5, clean_pdf_text("  - Normalizarea Datelor: Standardisation via StandardScaler (Z-Score Normalization)"), ln=1)
    pdf.cell(0, 5, clean_pdf_text("  - Set de Date Antrenament: UCI Cleveland Heart Disease Dataset"), ln=1)
    pdf.ln(5)
    
    # Sectiunea 3: Rezultate Comparative Algoritmi
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, clean_pdf_text("3. Rezultatele Comparative ale Algoritmilor (Benchmark)"), ln=1)
    pdf.set_font("Helvetica", "", 8)
    
    # Header Tabel
    pdf.set_fill_color(230, 230, 230)
    pdf.cell(55, 6, clean_pdf_text("Algorithm"), 1, 0, "C", True)
    pdf.cell(30, 6, clean_pdf_text("Accuracy"), 1, 0, "C", True)
    pdf.cell(30, 6, clean_pdf_text("Precision"), 1, 0, "C", True)
    pdf.cell(30, 6, clean_pdf_text("Recall"), 1, 0, "C", True)
    pdf.cell(35, 6, clean_pdf_text("F1 Score"), 1, 1, "C", True)
    
    # Linii Tabel
    for _, row in metrics_df.iterrows():
        pdf.cell(55, 5, clean_pdf_text(str(row['Algorithm'])), 1, 0, "L")
        pdf.cell(30, 5, f"{row['Accuracy']:.4f}", 1, 0, "C")
        pdf.cell(30, 5, f"{row['Precision']:.4f}", 1, 0, "C")
        pdf.cell(30, 5, f"{row['Recall']:.4f}", 1, 0, "C")
        pdf.cell(35, 5, f"{row['F1 Score']:.4f}", 1, 1, "C")
        
    pdf.ln(6)
    
    # Declinare a responsabilitatii
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, clean_pdf_text("AVIZ MEDICAL: Acest document este generat de un sistem software bazat pe inteligenta artificiala in scop educational si de asistenta a deciziilor. Nu reprezinta un diagnostic medical certificat si nu inlocuieste consultul medical de specialitate."))
    
    return bytes(pdf.output())


feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

tab1, tab2 = st.tabs(["Estimare Risc & Analiză SHAP", "Compararea algoritmilor"])

with tab1:
    st.header("Introducere Date Clinice Pacient")
    st.markdown("""
    Completați sau ajustați parametrii clinici ai pacientului în formularele de mai jos. 
    Aplicația recalculează în timp real riscul cardiovascular și nivelul de încredere asociat, permițând simulări ipotetice (*What-If scenarios*) pentru a observa modul în care modificarea unui singur factor (ex. scăderea colesterolului sau a tensiunii arteriale) influențează pronosticul final.
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input(
            "Vârstă", min_value=1, max_value=120, value=55,
            help="Vârsta pacientului exprimată în ani."
        )
        sex = st.selectbox(
            "Sex", options=[0, 1], format_func=lambda x: "Feminin" if x == 0 else "Masculin",
            help="Sexul biologic al pacientului (0 = Feminin, 1 = Masculin)."
        )
        cp = st.selectbox(
            "Tip Durere Toracică (CP)", [0, 1, 2, 3], 
            format_func=lambda x: ["Angină Tipică", "Angină Atipică", "Durere Neanginoasă", "Asimptomatic"][x],
            help="Tipul de disconfort toracic: Angină Tipică, Angină Atipică, Durere Neanginoasă sau Asimptomatic."
        )
        trestbps = st.number_input(
            "Tensiune Arterială în Repaus (mm Hg)", value=120,
            help="Tensiunea arterială în repaus (mm Hg) la internarea în spital. Valoarea normală este sub 120 mm Hg."
        )
        chol = st.number_input(
            "Colesterol Seric (mg/dl)", value=200,
            help="Nivelul colesterolului seric în mg/dl. Valorile peste 200 mg/dl indică risc crescut."
        )

    with col2:
        fbs = st.selectbox(
            "Glicemie în Jeun > 120 mg/dl", [0, 1], format_func=lambda x: "Nu" if x == 0 else "Da",
            help="Indică dacă glicemia pe nemâncate depășește 120 mg/dl (1 = Da, 0 = Nu)."
        )
        restecg = st.selectbox(
            "Rezultat EKG în Repaus", [0, 1, 2],
            format_func=lambda x: ["Normal (0)", "Anomalie Undă ST-T (1)", "Hipertrofie Ventriculară (2)"][x],
            help="Rezultatul electrocardiogramei în repaus: 0 = Normal, 1 = Anomalie de undă ST-T, 2 = Hipertrofie ventriculară stângă."
        )
        thalach = st.number_input(
            "Frecvență Cardiacă Maximă", value=150,
            help="Frecvența cardiacă maximă atinsă în timpul testului de efort."
        )
        exang = st.selectbox(
            "Angină Indusă de Efort", [0, 1], format_func=lambda x: "Nu" if x == 0 else "Da",
            help="Prezența durerii toracice în timpul efortului fizic (1 = Da, 0 = Nu)."
        )

    with col3:
        oldpeak = st.number_input(
            "Depresie ST (oldpeak)", value=1.0, step=0.1,
            help="Depresia ST indusă de efort raportată la starea de repaus."
        )
        slope = st.selectbox(
            "Panta Segmentului ST", [0, 1, 2],
            format_func=lambda x: ["Ascendentă (0)", "Orizontală (1)", "Descendentă (2)"][x],
            help="Panta segmentului ST de vârf în timpul efortului: 0 = Ascendentă, 1 = Orizontală, 2 = Descendentă."
        )
        ca = st.selectbox(
            "Număr Vase Majore (0-3)", [0, 1, 2, 3, 4],
            help="Numărul de vase sanguine majore colorate prin fluoroscopie."
        )
        thal = st.selectbox(
            "Talasemie / Perfuzie (thal)", [0, 1, 2, 3],
            format_func=lambda x: ["Normal (0)", "Defect Fix (1)", "Defect Reversibil (2)", "Altul (3)"][x],
            help="Rezultatul testului de perfuzie miocardică: 0 = Normal, 1 = Defect fix, 2 = Defect reversibil."
        )

    # --- CALCUL PREDICTIE IN TIMP REAL ---
    input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]], columns=feature_names)
    scaled_input = pd.DataFrame(scaler.transform(input_data), columns=feature_names)
    
    prediction = model.predict(scaled_input)[0]
    prob = model.predict_proba(scaled_input)[0][1] if hasattr(model, "predict_proba") else None

    c_vals = {
        "age": age, "sex": "Feminin" if sex == 0 else "Masculin",
        "cp": ["Angina Tipica", "Angina Atipica", "Durere Neanginoasa", "Asimptomatic"][cp],
        "trestbps": trestbps, "chol": chol, "fbs": "Da" if fbs == 1 else "Nu",
        "restecg": restecg, "thalach": thalach, "exang": "Da" if exang == 1 else "Nu",
        "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
    }

    st.markdown("---")
    
    # Sectiune evaluare rezultat
    st.subheader("Evaluarea Rezultatului Clinice")
    st.markdown("""
    Pe baza parametrilor introduși mai sus, modelul optimizat clasifică profilul pacientului și calculează o probabilitate estimativă a riscului de afecțiune cardiovasculară.
    """)
    
    risk_label = "RISC CRESCUT" if prediction == 1 else "RISC SCAZUT"

    if prob is not None:
        confidence_val = prob if prediction == 1 else (1 - prob)
        conf_str = f"{confidence_val * 100:.1f}%"
        risk_percentage = prob * 100
    else:
        conf_str = "N/A"
        risk_percentage = 0.0

    # Afisare banner rezultat
    col_res1, col_res2 = st.columns([3, 1])
    
    with col_res1:
        if prediction == 1:
            st.error(f"⚠️ **Rezultat Evaluare: Risc Crescut de Boală Cardiovasculară** (Încredere Model: {conf_str})")
        else:
            st.success(f"✅ **Rezultat Evaluare: Risc Scăzut de Boală Cardiovasculară** (Încredere Model: {conf_str})")
            
    with col_res2:
        st.metric(
            label="Scor Probabilitate Risc", 
            value=f"{risk_percentage:.1f}%", 
            delta="Risc Crescut" if prediction == 1 else "Risc Scăzut",
            delta_color="inverse" if prediction == 1 else "normal"
        )

    # --- Generare Raport PDF ---
    st.subheader("Generare Raport Clinic PDF")
    st.markdown("""
    Puteți descărca un raport clinic sumar în format PDF care conține fișa detaliată a pacientului, rezultatul evaluării de risc și tabelul comparativ al performanțelor algoritmilor.
    """)
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pdf_data = generate_pdf_report(c_vals, risk_label, conf_str, timestamp, metrics_df)

    # --- Buton Descarcare PDF ---
    st.download_button(
        label="📥 Descarcă Raport Pacient (.pdf)",
        data=pdf_data,
        file_name=f"raport_pacient_cardio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
        mime="application/pdf",
        type="secondary"
    )

    # --- Sectiunea Explicatie SHAP ---
    st.subheader("Explicarea Deciziei Modelului (Valori SHAP)")
    st.markdown("""
    Graficul de mai jos oferă o explicație individualizată a deciziei luate de model pentru acest pacient. 
    Utilizând cadrul matematic **SHAP (SHapley Additive exPlanations)**, fiecare parametru clinic este analizat pentru a determina în ce măsură a crescut (roșu) sau a scăzut (albastru) probabilitatea prezenței afecțiunii cardiovasculare.
    """)
    
    try:
        if hasattr(model, "tree_explanation_code") or "RandomForest" in str(type(model)) or "GradientBoosting" in str(type(model)):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer(scaled_input)
        else:
            explainer = shap.Explainer(model.predict, X_train_scaled)
            shap_values = explainer(scaled_input)

        if len(shap_values.shape) == 3:
            exp_to_plot = shap_values[0, :, 1]
        else:
            exp_to_plot = shap_values[0]
            
        display_inputs = [
            f"{c_vals['age']} ani", 
            c_vals['sex'], 
            c_vals['cp'], 
            f"{c_vals['trestbps']} mmHg", 
            f"{c_vals['chol']} mg/dl", 
            c_vals['fbs'], 
            f"EKG-{c_vals['restecg']}", 
            f"{c_vals['thalach']} bpm", 
            c_vals['exang'], 
            f"{c_vals['oldpeak']}", 
            f"Panta-{c_vals['slope']}", 
            f"{c_vals['ca']} vase", 
            f"Thal-{c_vals['thal']}"
        ]

        exp_to_plot.data = display_inputs

        fig, ax = plt.subplots(figsize=(8, 4))
        shap.plots.bar(exp_to_plot, max_display=14, show=False)
        plt.tight_layout()
        st.pyplot(fig)

        # --- Ghid de interpretare si dictionar clinic ---
        with st.expander("📖 **Ghid de Interpretare Grafic SHAP & Glosar Clinic**", expanded=True):
            guide_col1, guide_col2 = st.columns(2)

            with guide_col1:
                st.markdown("#### 📊 Interpretare Culori și Scoruri")
                st.markdown("""
                * 🔴 **Bara Roșie (Scor +):** Acest parametru a **crescut** scorul de risc al pacientului (a împins predicția spre **Risc Crescut**).
                * 🔵 **Bara Albastră (Scor -):** Acest parametru a **redus** scorul de risc al pacientului (a împins predicția spre **Risc Scăzut**).
                * 📏 **Lungimea Barei:** Indică magnitudinea influenței parametrului asupra deciziei finale.
                * 🔢 **Valori (ex: `-0.1`):** Reprezintă contribuția marginală exactă a parametrului la probabilitatea calculată.
                """)

            with guide_col2:
                st.markdown("#### 🩺 Ghid Parametri Clinici")
                st.markdown("""
                * **`cp` (Tip Durere Toracică):** 0: Angină Tipică, 1: Angină Atipică, 2: Durere Neanginoasă, 3: Asimptomatic.
                * **`thal` (Talasemie / Perfuzie):** Rezultat test perfuzie (0: Normal, 1: Defect fix, 2: Defect reversibil).
                * **`slope`:** Panta segmentului ST în efort.
                * **`ca`:** Număr de vase sanguine majore (0–4) vizibile la fluoroscopie.
                * **`exang`:** Angină indusă de efort fizic.
                * **`oldpeak`:** Depresia segmentului ST indusă de efort.
                * **`restecg`:** Rezultatul electrocardiogramei în repaus.
                * **`fbs`:** Glicemie în jeun > 120 mg/dl.
                * **`thalach`:** Frecvență cardiacă maximă atinsă.
                * **`trestbps`:** Tensiune arterială în repaus (mm Hg).
                * **`chol`:** Colesterol seric (mg/dl).
                """)

    except Exception as e:
        st.warning(f"Nu s-a putut genera graficul SHAP: {e}")

with tab2:
    st.header("Compararea performanței algoritmilor de Machine Learning")
    st.markdown("""
    Această secțiune prezintă o analiză comparativă riguroasă a celor 8 algoritmi de învățare automată evaluați pe setul de date clinic Cleveland. 
    Toate modelele au fost antrenate pe 80% din date și evaluate pe un set independent de test (20%), utilizând aceleași condiții de scalare și validare încrucișată.
    
    Metricile urmărite sunt:
    * **Accuracy (Acuratețe):** Proporția totală de predicții corecte.
    * **Precision (Precizie):** Capacitatea modelului de a nu eticheta un pacient sănătos ca fiind bolnav.
    * **Recall (Sensibilitate):** Capacitatea modelului de a identifica toți pacienții care suferă de o afecțiune cardiacă.
    * **F1 Score:** Media armonică dintre Precizie și Sensibilitate.
    """)
    
    st.dataframe(metrics_df, use_container_width=True)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    sns.barplot(data=metrics_df, x="Algorithm", y="Accuracy", ax=ax, palette="viridis")
    
    ax.set_ylim(0, 1.0)
    plt.xticks(rotation=45, ha='right')
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 5), textcoords='offset points')
    plt.tight_layout()
    st.pyplot(fig)