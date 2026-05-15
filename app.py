import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, f1_score, roc_auc_score
)
from sklearn.pipeline import Pipeline
import pickle
import os
import warnings

warnings.filterwarnings("ignore")

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}

/* Background */
.stApp {
    background: #0d0d0d;
    color: #f0ede8;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #141414;
    border-right: 1px solid #2a2a2a;
}

/* Metrics */
[data-testid="metric-container"] {
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 8px;
    padding: 16px;
}

/* Buttons */
.stButton > button {
    background: #e8ff4a;
    color: #0d0d0d;
    border: none;
    border-radius: 4px;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: #f5ff80;
    transform: translateY(-1px);
}

/* Text area */
.stTextArea > div > div > textarea {
    background: #1a1a1a;
    border: 1px solid #333;
    color: #f0ede8;
    font-family: 'DM Mono', monospace;
    border-radius: 6px;
}

/* Selectbox */
.stSelectbox > div > div {
    background: #1a1a1a;
    border: 1px solid #333;
    color: #f0ede8;
}

/* Header */
.main-header {
    text-align: center;
    padding: 40px 0 20px;
}
.main-header h1 {
    font-family: 'Syne', sans-serif;
    font-size: 3.5rem;
    font-weight: 800;
    color: #e8ff4a;
    letter-spacing: -0.02em;
    margin: 0;
}
.main-header p {
    color: #888;
    font-size: 1rem;
    margin-top: 8px;
    font-family: 'DM Mono', monospace;
}

/* Result cards */
.result-fake {
    background: linear-gradient(135deg, #3d0000, #1a0000);
    border: 1px solid #ff4444;
    border-radius: 12px;
    padding: 24px;
    text-align: center;
}
.result-real {
    background: linear-gradient(135deg, #003d1a, #001a0d);
    border: 1px solid #44ff88;
    border-radius: 12px;
    padding: 24px;
    text-align: center;
}
.result-label {
    font-size: 2.5rem;
    font-weight: 800;
    letter-spacing: -0.02em;
}
.result-fake .result-label { color: #ff6666; }
.result-real .result-label { color: #66ffaa; }

/* Section headers */
.section-title {
    font-size: 1.4rem;
    font-weight: 700;
    color: #e8ff4a;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    border-bottom: 1px solid #2a2a2a;
    padding-bottom: 8px;
    margin-bottom: 16px;
}

/* Info boxes */
.info-box {
    background: #1a1a1a;
    border-left: 3px solid #e8ff4a;
    padding: 12px 16px;
    border-radius: 0 6px 6px 0;
    margin: 8px 0;
    font-family: 'DM Mono', monospace;
    font-size: 0.85rem;
    color: #aaa;
}

/* Tabs */
.stTabs [data-baseweb="tab"] {
    color: #888;
    font-family: 'Syne', sans-serif;
    font-weight: 600;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    color: #e8ff4a;
    border-bottom-color: #e8ff4a;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border: 1px solid #2a2a2a;
    border-radius: 8px;
}

/* Progress bar */
.stProgress > div > div {
    background: #e8ff4a;
}

/* Spinner */
.stSpinner > div {
    border-top-color: #e8ff4a;
}
</style>
""", unsafe_allow_html=True)

# ── Label Mapping ─────────────────────────────────────────────────────────────
FAKE_LABELS = ['pants-fire', 'false', 'barely-true']
REAL_LABELS = ['half-true', 'mostly-true', 'true']

COLUMN_NAMES = [
    'id', 'label', 'statement', 'subjects', 'speaker',
    'speaker_job', 'state_info', 'party_affiliation',
    'barely_true_counts', 'false_counts', 'half_true_counts',
    'mostly_true_counts', 'pants_on_fire_counts', 'context'
]

# ── Helper Functions ──────────────────────────────────────────────────────────
def convert_label(label):
    return 0 if label in FAKE_LABELS else 1

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'www\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

@st.cache_resource(show_spinner=False)
def load_data(train_file, valid_file, test_file):
    """Load and preprocess LIAR dataset."""
    train_df = pd.read_table(train_file, header=None, names=COLUMN_NAMES)
    valid_df = pd.read_table(valid_file, header=None, names=COLUMN_NAMES)
    test_df  = pd.read_table(test_file,  header=None, names=COLUMN_NAMES)

    for df in [train_df, valid_df, test_df]:
        df['target']     = df['label'].apply(convert_label)
        df['clean_text'] = df['statement'].apply(clean_text)

    return train_df, valid_df, test_df

@st.cache_resource(show_spinner=False)
def train_models(train_df, test_df):
    """Train LR and GB models, return fitted pipelines + metrics."""
    vectorizer_lr = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X_train = vectorizer_lr.fit_transform(train_df['clean_text'])
    X_test  = vectorizer_lr.transform(test_df['clean_text'])
    y_train = train_df['target']
    y_test  = test_df['target']

    lr = LogisticRegression(max_iter=1000, class_weight='balanced')
    lr.fit(X_train, y_train)
    lr_preds = lr.predict(X_test)
    lr_proba = lr.predict_proba(X_test)[:, 1]

    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    gb.fit(X_train, y_train)
    gb_preds = gb.predict(X_test)
    gb_proba = gb.predict_proba(X_test)[:, 1]

    metrics = {
        "Logistic Regression": {
            "accuracy":  accuracy_score(y_test, lr_preds),
            "f1":        f1_score(y_test, lr_preds),
            "roc_auc":   roc_auc_score(y_test, lr_proba),
            "report":    classification_report(y_test, lr_preds,
                             target_names=["Fake", "Real"], output_dict=True),
            "cm":        confusion_matrix(y_test, lr_preds),
        },
        "Gradient Boosting": {
            "accuracy":  accuracy_score(y_test, gb_preds),
            "f1":        f1_score(y_test, gb_preds),
            "roc_auc":   roc_auc_score(y_test, gb_proba),
            "report":    classification_report(y_test, gb_preds,
                             target_names=["Fake", "Real"], output_dict=True),
            "cm":        confusion_matrix(y_test, gb_preds),
        },
    }

    return vectorizer_lr, lr, gb, metrics

def predict_single(text, vectorizer, model):
    cleaned = clean_text(text)
    vec     = vectorizer.transform([cleaned])
    pred    = model.predict(vec)[0]
    proba   = model.predict_proba(vec)[0]
    return pred, proba

def plot_confusion_matrix(cm, title, ax):
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='YlOrRd',
        xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'],
        ax=ax, linewidths=0.5, linecolor='#333',
        annot_kws={"size": 14, "weight": "bold"}
    )
    ax.set_title(title, fontsize=12, fontweight='bold', color='#e8ff4a', pad=10)
    ax.set_xlabel('Predicted', color='#aaa')
    ax.set_ylabel('Actual', color='#aaa')
    ax.tick_params(colors='#aaa')
    ax.figure.patch.set_facecolor('#0d0d0d')
    ax.set_facecolor('#1a1a1a')

def generate_wordcloud(text_series, bg_color='#0d0d0d'):
    text = ' '.join(text_series)
    wc = WordCloud(
        width=800, height=400,
        background_color=bg_color,
        colormap='YlOrBr',
        max_words=150,
    ).generate(text)
    return wc

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    st.markdown("### 📂 Upload Dataset")
    train_file = st.file_uploader("train.tsv", type=["tsv"])
    valid_file = st.file_uploader("valid.tsv", type=["tsv"])
    test_file  = st.file_uploader("test.tsv",  type=["tsv"])

    use_sample = False
    if not (train_file and valid_file and test_file):
        st.markdown('<div class="info-box">Upload the LIAR dataset files, or use sample data below.</div>',
                    unsafe_allow_html=True)
        use_sample = st.checkbox("Use built-in sample data", value=True)

    st.markdown("---")
    st.markdown("### 🤖 Model")
    model_choice = st.selectbox(
        "Choose classifier",
        ["Logistic Regression", "Gradient Boosting"]
    )

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
<div class="info-box">
LIAR dataset · 12,836 statements<br>
Binary: Fake (0) vs Real (1)<br>
Features: TF-IDF (10K, 1-2 grams)
</div>
""", unsafe_allow_html=True)

# ── Main Header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🔍 FAKE NEWS DETECTOR</h1>
    <p>LIAR Dataset · TF-IDF + Logistic Regression / Gradient Boosting · Binary Classification</p>
</div>
""", unsafe_allow_html=True)

# ── Load / Generate Data ──────────────────────────────────────────────────────
if use_sample or (train_file and valid_file and test_file):

    with st.spinner("Loading & preprocessing data…"):
        if use_sample:
            # Minimal synthetic dataset for demo
            rng = np.random.default_rng(42)
            fake_statements = [
                "Scientists discover moon is made of cheese",
                "Vaccines contain microchips to track citizens",
                "The earth is flat according to NASA insiders",
                "President secretly signs bill to ban cars",
                "Local man wins lottery seven times in a row using secret formula",
                "Aliens landed in New Mexico and are working for the government",
                "Drinking bleach cures all diseases according to doctors",
                "The sun is getting colder every year secret reports reveal",
            ] * 50
            real_statements = [
                "Congress passed the infrastructure bill last Tuesday",
                "The unemployment rate dropped to 3.8 percent in March",
                "FDA approves new drug for treatment of diabetes",
                "Federal Reserve raised interest rates by 25 basis points",
                "Scientists confirm CO2 levels reached record high last year",
                "The Supreme Court ruled on gerrymandering case in a 5-4 decision",
                "State governors met to discuss education funding reforms",
                "New study links moderate exercise to improved cardiovascular health",
            ] * 50

            all_statements = fake_statements + real_statements
            all_labels_raw = ['false'] * len(fake_statements) + ['true'] * len(real_statements)
            idx = rng.permutation(len(all_statements))
            all_statements = [all_statements[i] for i in idx]
            all_labels_raw = [all_labels_raw[i] for i in idx]

            full_df = pd.DataFrame({
                'id': range(len(all_statements)),
                'label': all_labels_raw,
                'statement': all_statements,
                'subjects': 'general',
                'speaker': 'unknown',
                'speaker_job': 'unknown',
                'state_info': 'unknown',
                'party_affiliation': 'none',
                'barely_true_counts': 0,
                'false_counts': 0,
                'half_true_counts': 0,
                'mostly_true_counts': 0,
                'pants_on_fire_counts': 0,
                'context': 'unknown',
            })
            full_df['target']     = full_df['label'].apply(convert_label)
            full_df['clean_text'] = full_df['statement'].apply(clean_text)

            split1 = int(0.7 * len(full_df))
            split2 = int(0.85 * len(full_df))
            train_df = full_df.iloc[:split1].reset_index(drop=True)
            valid_df = full_df.iloc[split1:split2].reset_index(drop=True)
            test_df  = full_df.iloc[split2:].reset_index(drop=True)
        else:
            train_df, valid_df, test_df = load_data(train_file, valid_file, test_file)

    with st.spinner("Training models…"):
        vectorizer, lr_model, gb_model, metrics = train_models(train_df, test_df)

    selected_model = lr_model if model_choice == "Logistic Regression" else gb_model

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔎 Predict",
        "📊 Model Performance",
        "📈 Data Exploration",
        "📋 Dataset Preview"
    ])

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 1 · PREDICT
    # ─────────────────────────────────────────────────────────────────────────
    with tab1:
        st.markdown('<div class="section-title">Enter a Statement to Analyze</div>',
                    unsafe_allow_html=True)

        example_texts = [
            "Select an example…",
            "The unemployment rate dropped to 3.8 percent last month.",
            "Scientists secretly discovered that the moon controls the weather on earth.",
            "Congress passed a new bill to increase infrastructure spending.",
            "A secret government program is turning birds into surveillance drones.",
        ]
        example = st.selectbox("Try an example", example_texts)

        user_input = st.text_area(
            "Statement",
            value="" if example == example_texts[0] else example,
            height=120,
            placeholder="Type or paste a news statement here…",
            label_visibility="collapsed"
        )

        col_btn, col_clear = st.columns([1, 5])
        with col_btn:
            predict_btn = st.button("🔍 ANALYZE", width='stretch')

        if predict_btn and user_input.strip():
            pred, proba = predict_single(user_input, vectorizer, selected_model)

            st.markdown("<br>", unsafe_allow_html=True)
            col_res, col_conf = st.columns([1, 1])

            with col_res:
                if pred == 0:
                    st.markdown(f"""
<div class="result-fake">
    <div style="font-size:2rem;margin-bottom:8px">⚠️</div>
    <div class="result-label">FAKE</div>
    <div style="color:#ff9999;margin-top:8px;font-family:'DM Mono',monospace;font-size:0.85rem">
        This statement is likely misinformation
    </div>
</div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
<div class="result-real">
    <div style="font-size:2rem;margin-bottom:8px">✅</div>
    <div class="result-label">REAL</div>
    <div style="color:#99ffcc;margin-top:8px;font-family:'DM Mono',monospace;font-size:0.85rem">
        This statement appears credible
    </div>
</div>""", unsafe_allow_html=True)

            with col_conf:
                st.markdown('<div class="section-title" style="font-size:1rem">Confidence Scores</div>',
                            unsafe_allow_html=True)
                st.markdown(f"**Fake probability:** `{proba[0]:.1%}`")
                st.progress(float(proba[0]))
                st.markdown(f"**Real probability:** `{proba[1]:.1%}`")
                st.progress(float(proba[1]))

                st.markdown(f"""
<div class="info-box" style="margin-top:12px">
Model: {model_choice}<br>
Confidence: {"HIGH" if max(proba) > 0.75 else "MEDIUM" if max(proba) > 0.55 else "LOW"} ({max(proba):.1%})
</div>""", unsafe_allow_html=True)

        elif predict_btn:
            st.warning("Please enter a statement to analyze.")

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 2 · MODEL PERFORMANCE
    # ─────────────────────────────────────────────────────────────────────────
    with tab2:
        st.markdown('<div class="section-title">Performance Metrics</div>',
                    unsafe_allow_html=True)

        # Summary table
        summary_data = []
        for name, m in metrics.items():
            summary_data.append({
                "Model": name,
                "Accuracy": f"{m['accuracy']:.3f}",
                "F1 Score": f"{m['f1']:.3f}",
                "ROC-AUC":  f"{m['roc_auc']:.3f}",
            })
        st.dataframe(pd.DataFrame(summary_data), width='stretch', hide_index=True)

        # Metric cards for selected model
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f'<div class="section-title">{model_choice} · Detailed Metrics</div>',
                    unsafe_allow_html=True)

        m = metrics[model_choice]
        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy", f"{m['accuracy']:.3f}")
        c2.metric("F1 Score", f"{m['f1']:.3f}")
        c3.metric("ROC-AUC",  f"{m['roc_auc']:.3f}")

        # Confusion matrices
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">Confusion Matrices</div>',
                    unsafe_allow_html=True)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        fig.patch.set_facecolor('#0d0d0d')
        for ax, (name, m_data) in zip(axes, metrics.items()):
            plot_confusion_matrix(m_data['cm'], name, ax)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Classification report
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f'<div class="section-title">{model_choice} · Classification Report</div>',
                    unsafe_allow_html=True)

        report = metrics[model_choice]['report']
        report_rows = []
        for cls in ['Fake', 'Real', 'macro avg', 'weighted avg']:
            key = cls.lower()
            if key in report:
                row = report[key]
                report_rows.append({
                    "Class":     cls,
                    "Precision": f"{row['precision']:.3f}",
                    "Recall":    f"{row['recall']:.3f}",
                    "F1-Score":  f"{row['f1-score']:.3f}",
                    "Support":   int(row['support']),
                })
        st.dataframe(pd.DataFrame(report_rows), width='stretch', hide_index=True)

        # Bar chart comparison
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">Model Comparison</div>',
                    unsafe_allow_html=True)

        model_names = list(metrics.keys())
        acc_vals = [metrics[n]['accuracy'] for n in model_names]
        f1_vals  = [metrics[n]['f1']       for n in model_names]
        auc_vals = [metrics[n]['roc_auc']  for n in model_names]

        x = np.arange(len(model_names))
        width = 0.25

        fig2, ax2 = plt.subplots(figsize=(8, 4))
        fig2.patch.set_facecolor('#0d0d0d')
        ax2.set_facecolor('#1a1a1a')
        ax2.bar(x - width, acc_vals, width, label='Accuracy', color='#e8ff4a', alpha=0.9)
        ax2.bar(x,         f1_vals,  width, label='F1 Score', color='#44ddff', alpha=0.9)
        ax2.bar(x + width, auc_vals, width, label='ROC-AUC',  color='#ff6688', alpha=0.9)
        ax2.set_xticks(x)
        ax2.set_xticklabels(model_names, color='#aaa')
        ax2.set_ylim(0, 1.1)
        ax2.set_ylabel('Score', color='#aaa')
        ax2.tick_params(colors='#aaa')
        ax2.legend(labelcolor='#aaa', facecolor='#1a1a1a', edgecolor='#333')
        ax2.grid(axis='y', alpha=0.2, color='#444')
        for spine in ax2.spines.values():
            spine.set_edgecolor('#333')
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 3 · DATA EXPLORATION
    # ─────────────────────────────────────────────────────────────────────────
    with tab3:
        st.markdown('<div class="section-title">Label Distribution</div>',
                    unsafe_allow_html=True)

        col_a, col_b = st.columns(2)

        with col_a:
            label_counts = train_df['label'].value_counts()
            fig3, ax3 = plt.subplots(figsize=(5, 4))
            fig3.patch.set_facecolor('#0d0d0d')
            ax3.set_facecolor('#1a1a1a')
            colors = ['#ff6688' if l in FAKE_LABELS else '#44ddff'
                      for l in label_counts.index]
            bars = ax3.bar(label_counts.index, label_counts.values,
                           color=colors, alpha=0.9, edgecolor='#333')
            ax3.set_xticklabels(label_counts.index, rotation=30, ha='right', color='#aaa', fontsize=9)
            ax3.set_ylabel('Count', color='#aaa')
            ax3.set_title('Original 6-Class Distribution', color='#e8ff4a', fontweight='bold')
            ax3.tick_params(colors='#aaa')
            ax3.grid(axis='y', alpha=0.2, color='#444')
            for spine in ax3.spines.values():
                spine.set_edgecolor('#333')
            plt.tight_layout()
            st.pyplot(fig3)
            plt.close()

        with col_b:
            binary_counts = train_df['target'].value_counts()
            fig4, ax4 = plt.subplots(figsize=(5, 4))
            fig4.patch.set_facecolor('#0d0d0d')
            ax4.set_facecolor('#1a1a1a')
            ax4.pie(
                binary_counts.values,
                labels=['Real' if i == 1 else 'Fake' for i in binary_counts.index],
                colors=['#44ddff', '#ff6688'],
                autopct='%1.1f%%',
                startangle=90,
                textprops={'color': '#f0ede8', 'fontsize': 12}
            )
            ax4.set_title('Binary Label Split', color='#e8ff4a', fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig4)
            plt.close()

        # Word Clouds
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">Word Clouds</div>',
                    unsafe_allow_html=True)

        col_c, col_d = st.columns(2)

        with col_c:
            st.markdown("**🔴 Fake News**")
            fake_texts = train_df[train_df['target'] == 0]['clean_text']
            if len(fake_texts) > 0:
                wc_fake = generate_wordcloud(fake_texts)
                fig5, ax5 = plt.subplots(figsize=(6, 3))
                fig5.patch.set_facecolor('#0d0d0d')
                ax5.imshow(wc_fake, interpolation='bilinear')
                ax5.axis('off')
                plt.tight_layout()
                st.pyplot(fig5)
                plt.close()

        with col_d:
            st.markdown("**🔵 Real News**")
            real_texts = train_df[train_df['target'] == 1]['clean_text']
            if len(real_texts) > 0:
                wc_real = WordCloud(
                    width=800, height=400,
                    background_color='#0d0d0d',
                    colormap='winter',
                    max_words=150,
                ).generate(' '.join(real_texts))
                fig6, ax6 = plt.subplots(figsize=(6, 3))
                fig6.patch.set_facecolor('#0d0d0d')
                ax6.imshow(wc_real, interpolation='bilinear')
                ax6.axis('off')
                plt.tight_layout()
                st.pyplot(fig6)
                plt.close()

        # Statement length distribution
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">Statement Length Distribution</div>',
                    unsafe_allow_html=True)

        train_df['word_count'] = train_df['statement'].apply(lambda x: len(str(x).split()))

        fig7, ax7 = plt.subplots(figsize=(10, 4))
        fig7.patch.set_facecolor('#0d0d0d')
        ax7.set_facecolor('#1a1a1a')
        ax7.hist(
            train_df[train_df['target'] == 0]['word_count'],
            bins=40, alpha=0.7, color='#ff6688', label='Fake', density=True
        )
        ax7.hist(
            train_df[train_df['target'] == 1]['word_count'],
            bins=40, alpha=0.7, color='#44ddff', label='Real', density=True
        )
        ax7.set_xlabel('Word Count', color='#aaa')
        ax7.set_ylabel('Density', color='#aaa')
        ax7.set_title('Statement Word Count by Class', color='#e8ff4a', fontweight='bold')
        ax7.tick_params(colors='#aaa')
        ax7.legend(labelcolor='#aaa', facecolor='#1a1a1a', edgecolor='#333')
        ax7.grid(alpha=0.2, color='#444')
        for spine in ax7.spines.values():
            spine.set_edgecolor('#333')
        plt.tight_layout()
        st.pyplot(fig7)
        plt.close()

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 4 · DATASET PREVIEW
    # ─────────────────────────────────────────────────────────────────────────
    with tab4:
        st.markdown('<div class="section-title">Dataset Overview</div>',
                    unsafe_allow_html=True)

        col_s1, col_s2, col_s3 = st.columns(3)
        col_s1.metric("Train Samples", f"{len(train_df):,}")
        col_s2.metric("Valid Samples", f"{len(valid_df):,}")
        col_s3.metric("Test Samples",  f"{len(test_df):,}")

        st.markdown("<br>", unsafe_allow_html=True)
        split_view = st.radio("View split:", ["Train", "Validation", "Test"], horizontal=True)
        df_map = {"Train": train_df, "Validation": valid_df, "Test": test_df}
        df_show = df_map[split_view]

        display_cols = ['statement', 'label', 'target', 'speaker', 'subjects']
        display_cols = [c for c in display_cols if c in df_show.columns]
        st.dataframe(
            df_show[display_cols].rename(columns={'target': 'binary_label'}),
            width='stretch',
            height=400
        )

else:
    # ── Landing / no data ─────────────────────────────────────────────────────
    st.markdown("<br><br>", unsafe_allow_html=True)
    col_info, col_steps = st.columns([1, 1])
    with col_info:
        st.markdown("""
<div style="background:#1a1a1a;border:1px solid #2a2a2a;border-radius:12px;padding:28px">
    <div style="color:#e8ff4a;font-size:1.2rem;font-weight:700;margin-bottom:16px">
        📦 What you need
    </div>
    <div class="info-box">Upload <b>train.tsv</b>, <b>valid.tsv</b>, <b>test.tsv</b><br>
    from the LIAR dataset in the sidebar.<br><br>
    Or enable <b>"Use built-in sample data"</b> to demo instantly.</div>
</div>
""", unsafe_allow_html=True)
    with col_steps:
        st.markdown("""
<div style="background:#1a1a1a;border:1px solid #2a2a2a;border-radius:12px;padding:28px">
    <div style="color:#e8ff4a;font-size:1.2rem;font-weight:700;margin-bottom:16px">
        🎯 Features
    </div>
    <div class="info-box">
    ✔ Live fake/real prediction<br>
    ✔ Confidence scores<br>
    ✔ Model performance metrics<br>
    ✔ Confusion matrices<br>
    ✔ Word clouds & EDA<br>
    ✔ LR & Gradient Boosting
    </div>
</div>
""", unsafe_allow_html=True)