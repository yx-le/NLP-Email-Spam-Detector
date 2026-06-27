import html
import os
import re
from pathlib import Path

# Streamlit's file watcher can inspect torch.classes as if it were a normal
# Python package path, which raises a PyTorch runtime error during BERT reruns.
os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")

import joblib
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from preprocessing import preprocess_indonesia_text, preprocess_text


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR.parent / "dataset"

# Change this after comparing your models:
# "logistic_regression", "naive_bayes", or "bert"
MODEL_TYPE = os.getenv("SPAM_MODEL_TYPE", "bert")

MODEL_FILES = {
    "logistic_regression": MODEL_DIR / "logistic_regression_model.joblib",
    "naive_bayes": MODEL_DIR / "naive_bayes_model.joblib",
    "bert": MODEL_DIR / "bert_spam_model",
}

INDONESIA_MODEL_FILES = {
    "bert": MODEL_DIR / "bert_indonesia_spam_model",
}

INDONESIA_MODEL_TYPE = "bert"


st.set_page_config(
    page_title="Bilingual Email Spam Detector",
    page_icon="E",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --app-teal: #159587;
        --app-coral: #e76f51;
    }
    .block-container {
        padding-top: 5.25rem;
        padding-bottom: 3rem;
        max-width: 1280px;
    }
    h1, h2, h3 {color: var(--text-color); letter-spacing: 0;}
    [data-testid="stHeader"] {
        background: color-mix(in srgb, var(--background-color) 88%, transparent);
        backdrop-filter: blur(10px);
    }
    [data-testid="stHeaderActionElements"] a {display: none !important;}
    [data-testid="stApp"]:has([data-testid="stSidebar"][aria-expanded="true"])
    [data-testid="stSidebarCollapsedControl"] {display: none !important;}
    div:has(> [data-testid="stVegaLiteChart"]) > [data-testid="stElementToolbar"] {
        display: none !important;
    }
    [data-testid="stVegaLiteChart"] {
        pointer-events: none !important;
        touch-action: pan-y !important;
    }
    [data-testid="stSidebar"] {background: var(--secondary-background-color);}
    [data-testid="stSidebar"] hr {border-color: color-mix(in srgb, var(--text-color) 16%, transparent);}
    [data-testid="stMetric"] {
        border: 1px solid color-mix(in srgb, var(--text-color) 14%, transparent);
        border-left: 4px solid var(--app-teal);
        padding: .75rem 1rem;
        background: var(--secondary-background-color);
        min-height: 118px;
    }
    .app-kicker {
        color: var(--app-teal);
        font-weight: 700;
        font-size: .78rem;
        text-transform: uppercase;
    }
    .app-summary {
        font-size: 1.05rem;
        max-width: 820px;
        color: color-mix(in srgb, var(--text-color) 74%, transparent);
    }
    .step-number {
        display: inline-block;
        width: 1.8rem;
        height: 1.8rem;
        line-height: 1.8rem;
        text-align: center;
        color: white;
        background: var(--app-coral);
        border-radius: 50%;
        margin-right: .45rem;
        font-weight: 700;
    }
    div[data-testid="stButton"] button,
    div[data-testid="stDownloadButton"] button {
        min-height: 2.65rem;
    }
    div[data-testid="stDataFrame"] {border-radius: 6px; overflow: hidden;}
    @media (max-width: 768px) {
        .block-container {padding-top: 4.75rem; padding-left: 1rem; padding-right: 1rem;}
        h1 {font-size: 2rem;}
    }
    div[data-testid="stImage"] img {
        background: transparent;
        filter: drop-shadow(0 0 .45px var(--text-color));
    }
    .confusion-grid {
        display: grid;
        grid-template-columns: minmax(86px, .85fr) repeat(2, 1fr);
        gap: 6px;
        align-items: stretch;
        margin: .5rem 0 1rem;
    }
    .confusion-label {
        display: flex;
        align-items: center;
        justify-content: center;
        min-height: 46px;
        padding: .4rem;
        color: color-mix(in srgb, var(--text-color) 78%, transparent);
        font-size: .86rem;
        font-weight: 600;
        text-align: center;
    }
    .confusion-cell {
        display: flex;
        min-height: 104px;
        align-items: center;
        justify-content: center;
        border: 1px solid color-mix(in srgb, var(--text-color) 12%, transparent);
        border-radius: 4px;
        color: var(--text-color);
        font-size: 1.7rem;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

PAGES = [
    "Home / About",
    "Text Analyzer",
    "Data Explorer",
    "Visualizations",
    "Model Info",
]

TEAM_MEMBERS = [
    "Lai Yan Qi",
    "Chua Le Jun",
    "Woo Yong Kang",
    "Le Yong Xiang",
]

TRANSLATIONS = {
    "English": {
        "section_label": "Live classification",
        "title": "Email Spam Detector",
        "description": (
            "Paste an English email below to classify it as legitimate "
            "(ham) or spam."
        ),
        "model": "Deployed model",
        "language_notice": (
            "The interface supports English and Bahasa Indonesia. The model "
            "shown here is trained on English emails."
        ),
        "email_content": "Email content",
        "placeholder": "Paste the email subject and message here...",
        "try_example": "Try an example",
        "load_example": "Load example",
        "clear": "Clear",
        "analyze": "Analyze Email",
        "loading": "Loading model and analyzing the email...",
        "explaining": "Calculating word influence with SHAP...",
        "changed": (
            "The email text has changed. Select Analyze Email to refresh "
            "the result."
        ),
        "empty": "Please enter some email content.",
        "prediction": "Prediction",
        "confidence": "Confidence",
        "ham": "HAM",
        "spam": "SPAM",
        "class_probabilities": "Class Probabilities",
        "class": "Class",
        "probability": "Probability",
        "ham_chart": "Ham",
        "spam_chart": "Spam",
        "indicators": "Spam Indicator Words",
        "explanation_title": "Why the model made this prediction",
        "spam_support": "Increases spam score",
        "ham_support": "Reduces spam score",
        "explanation_note": (
            "SHAP estimates how each token changes the spam prediction. "
            "Positive values increase the spam score and negative values "
            "reduce it. The final prediction uses all tokens together."
        ),
        "no_explanation": (
            "No individual word had a strong measurable effect on this "
            "prediction."
        ),
        "explanation_error": (
            "The prediction completed, but the word influence explanation "
            "could not be generated in this environment."
        ),
        "word_phrase": "Word or phrase",
        "influence": "Influence",
        "impact_score": "Impact score",
        "classic_note": (
            "TF-IDF impact combines each present feature value with its "
            "learned model weight."
        ),
        "bert_explanation": (
            "BERT does not provide reliable indicator words from simple "
            "coefficients. An explanation method such as SHAP can be added."
        ),
        "processed": "Text used by the model",
        "error": "Could not run the model",
        "performance": "Model Performance",
    },
    "Bahasa Indonesia": {
        "section_label": "Klasifikasi masa nyata",
        "title": "Pengesan E-mel Spam",
        "description": (
            "Tempel email Bahasa Indonesia untuk mengklasifikasikannya "
            "sebagai e-mel sah atau spam."
        ),
        "model": "Model digunakan",
        "language_notice": (
            "Pilihan Bahasa Indonesia menggunakan model yang dilatih khusus "
            "menggunakan dataset email Bahasa Indonesia."
        ),
        "email_content": "Kandungan e-mel",
        "placeholder": "Tampal subjek dan mesej e-mel di sini...",
        "try_example": "Cuba contoh",
        "load_example": "Muatkan contoh",
        "clear": "Kosongkan",
        "analyze": "Analisis E-mel",
        "loading": "Memuatkan model dan menganalisis e-mel...",
        "explaining": "Mengira pengaruh perkataan menggunakan SHAP...",
        "changed": (
            "Teks e-mel telah diubah. Pilih Analisis E-mel untuk mengemas "
            "kini keputusan."
        ),
        "empty": "Sila masukkan kandungan e-mel.",
        "prediction": "Ramalan",
        "confidence": "Tahap keyakinan",
        "ham": "BUKAN SPAM",
        "spam": "SPAM",
        "class_probabilities": "Kebarangkalian Kelas",
        "class": "Kelas",
        "probability": "Kebarangkalian",
        "ham_chart": "Bukan Spam",
        "spam_chart": "Spam",
        "indicators": "Perkataan Petunjuk Spam",
        "explanation_title": "Mengapa model membuat ramalan ini",
        "spam_support": "Meningkatkan skor spam",
        "ham_support": "Mengurangkan skor spam",
        "explanation_note": (
            "SHAP menganggarkan kesan setiap token terhadap ramalan spam. "
            "Nilai positif meningkatkan skor spam dan nilai negatif "
            "mengurangkannya. Ramalan akhir menggunakan semua token bersama."
        ),
        "no_explanation": (
            "Tiada perkataan individu yang memberi kesan kuat terhadap "
            "ramalan ini."
        ),
        "explanation_error": (
            "Prediksi selesai, tetapi penjelasan pengaruh kata tidak dapat "
            "dibuat dalam environment ini."
        ),
        "word_phrase": "Perkataan atau frasa",
        "influence": "Pengaruh",
        "impact_score": "Skor impak",
        "classic_note": (
            "Impak TF-IDF menggabungkan nilai setiap ciri yang hadir dengan "
            "pemberat model yang telah dipelajari."
        ),
        "bert_explanation": (
            "BERT tidak memberikan perkataan petunjuk yang boleh dipercayai "
            "melalui pekali mudah. Kaedah seperti SHAP boleh ditambah."
        ),
        "processed": "Teks yang digunakan oleh model",
        "error": "Model tidak dapat dijalankan",
        "performance": "Prestasi Model",
    },
}


def format_model_name(model_type):
    return {
        "bert": "BERT",
        "logistic_regression": "Logistic Regression",
        "naive_bayes": "Multinomial Naive Bayes",
    }.get(model_type, model_type.replace("_", " ").title())


def get_language_model_config(language):
    def bert_checkpoint_is_valid(model_dir):
        if not model_dir.is_dir():
            return False

        weights_path = model_dir / "model.safetensors"
        if not weights_path.is_file() or weights_path.stat().st_size < 1024:
            return False

        with weights_path.open("rb") as weights_file:
            header_size_bytes = weights_file.read(8)

        if len(header_size_bytes) != 8:
            return False

        header_size = int.from_bytes(header_size_bytes, byteorder="little")
        return 0 < header_size < weights_path.stat().st_size - 8

    if language == "Bahasa Indonesia":
        model_type = INDONESIA_MODEL_TYPE
        model_files = INDONESIA_MODEL_FILES
        return model_type, model_files[model_type], None

    model_type = MODEL_TYPE
    vectorizer_path = MODEL_DIR / "tfidf_vectorizer.joblib"
    if (
        model_type == "bert"
        and not bert_checkpoint_is_valid(MODEL_FILES["bert"])
        and MODEL_FILES["logistic_regression"].exists()
        and vectorizer_path.exists()
    ):
        model_type = "logistic_regression"

    return model_type, MODEL_FILES[model_type], vectorizer_path


@st.cache_resource
def load_model_resources(model_type, model_path, vectorizer_path):
    if model_type in {"logistic_regression", "naive_bayes"}:
        if not vectorizer_path.exists() or not model_path.exists():
            raise FileNotFoundError(
                "Required Joblib model files were not found in models/."
            )

        return {
            "model": joblib.load(model_path),
            "vectorizer": joblib.load(vectorizer_path),
        }

    if model_type == "bert":
        try:
            import torch
            from transformers import AutoTokenizer, BertForSequenceClassification
        except Exception as error:
            raise RuntimeError(
                "BERT dependencies are not installed correctly. Reinstall the "
                "app requirements inside .venv, especially torch, torchvision, "
                "transformers, tokenizers, and safetensors."
            ) from error

        patch_torch_classes_path(torch)

        if not model_path.exists():
            raise FileNotFoundError(
                "The BERT model directory was not found in models/."
            )

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = BertForSequenceClassification.from_pretrained(model_path)
        model.eval()

        return {
            "model": model,
            "tokenizer": tokenizer,
            "torch": torch,
        }

    raise ValueError(f"Unsupported model type: {model_type}")


def patch_torch_classes_path(torch_module):
    class TorchClassesPath:
        _path = []

    try:
        torch_module.classes.__path__ = TorchClassesPath()
    except Exception:
        pass


def predict_classic(email_text, resources, language):
    if language == "Bahasa Indonesia":
        cleaned_text = preprocess_indonesia_text(email_text)
    else:
        cleaned_text = preprocess_text(email_text)
    features = resources["vectorizer"].transform([cleaned_text])
    probabilities = resources["model"].predict_proba(features)[0]
    prediction = int(np.argmax(probabilities))

    indicators = get_classic_indicators(
        resources["model"],
        resources["vectorizer"],
        features,
    )

    return prediction, probabilities, cleaned_text, indicators


def get_classic_indicators(model, vectorizer, features, limit=12):
    feature_names = vectorizer.get_feature_names_out()

    if hasattr(model, "coef_"):
        spam_weights = model.coef_[0]
    elif hasattr(model, "feature_log_prob_"):
        spam_weights = (
            model.feature_log_prob_[1] - model.feature_log_prob_[0]
        )
    else:
        return []

    row = features.getrow(0)
    contributions = row.data * spam_weights[row.indices]
    ranked_indices = np.argsort(np.abs(contributions))[::-1]

    indicators = []
    for position in ranked_indices:
        if abs(contributions[position]) < 0.000001:
            continue

        phrase = feature_names[row.indices[position]]
        if phrase not in {item["word"] for item in indicators}:
            indicators.append(
                {
                    "word": phrase,
                    "impact": float(contributions[position]),
                }
            )

        if len(indicators) == limit:
            break

    return indicators


def predict_bert(email_text, resources):
    torch = resources["torch"]
    encoded = resources["tokenizer"](
        email_text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
        padding=True,
    )

    with torch.no_grad():
        logits = resources["model"](**encoded).logits
        probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()

    spam_probability = float(probabilities[1])
    SPAM_THRESHOLD = 0.70
    prediction = 1 if spam_probability >= SPAM_THRESHOLD else 0
    return prediction, probabilities, email_text, []


@st.cache_resource(show_spinner=False)
def create_shap_explainer(_model, _tokenizer, _torch):
    import shap

    def predict_probabilities(texts):
        text_list = [str(value) for value in list(texts)]
        encoded = _tokenizer(
            text_list,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=True,
        )
        encoded = {
            key: value.to(_model.device)
            for key, value in encoded.items()
        }

        with _torch.no_grad():
            logits = _model(**encoded).logits
            return _torch.softmax(logits, dim=1).cpu().numpy()

    return shap.Explainer(
        predict_probabilities,
        _tokenizer,
        output_names=["HAM", "SPAM"],
        algorithm="partition",
    )


def explain_bert_words(email_text, resources, result_limit=15):
    explanation_token_ids = resources["tokenizer"].encode(
        email_text,
        add_special_tokens=False,
        truncation=True,
        max_length=128,
    )
    explanation_text = resources["tokenizer"].decode(
        explanation_token_ids,
        skip_special_tokens=True,
    )

    explainer = create_shap_explainer(
        resources["model"],
        resources["tokenizer"],
        resources["torch"],
    )

    explanation = explainer(
        [explanation_text],
        max_evals=300,
        batch_size=8,
    )[0]

    values = np.asarray(explanation.values)
    tokens = np.asarray(explanation.data)

    if values.ndim == 2:
        spam_values = values[:, 1]
    else:
        spam_values = values

    aggregated = {}
    display_names = {}

    for token, value in zip(tokens, spam_values):
        clean_token = str(token).replace("##", "").strip()
        if (
            not clean_token
            or clean_token in {"[CLS]", "[SEP]", "[PAD]"}
            or not re.search(r"[A-Za-z0-9]", clean_token)
        ):
            continue

        normalized = clean_token.lower()
        aggregated[normalized] = aggregated.get(normalized, 0.0) + float(value)
        display_names.setdefault(normalized, clean_token)

    impacts = [
        {
            "word": display_names[token],
            "impact": impact,
        }
        for token, impact in aggregated.items()
        if abs(impact) >= 0.000001
    ]

    return sorted(
        impacts,
        key=lambda item: abs(item["impact"]),
        reverse=True,
    )[:result_limit]


def highlight_bert_impacts(text_value, impacts):
    safe_text = html.escape(text_value)
    impact_map = {
        item["word"].lower(): item["impact"]
        for item in impacts
    }

    if not impact_map:
        return safe_text.replace("\n", "<br>")

    words = sorted(impact_map, key=len, reverse=True)
    pattern = re.compile(
        r"\b(" + "|".join(re.escape(word) for word in words) + r")\b",
        flags=re.IGNORECASE,
    )

    def replace_word(match):
        word = match.group(0)
        impact = impact_map[word.lower()]
        color = "#ffb3b3" if impact > 0 else "#b8f2c2"
        return (
            f"<mark style='background-color:{color};"
            f"color:#18201e;padding:2px 3px;border-radius:3px'>{word}</mark>"
        )

    return pattern.sub(replace_word, safe_text).replace("\n", "<br>")


def highlight_indicators(text, indicators):
    safe_text = html.escape(text)
    words = sorted(
        {part for phrase in indicators for part in phrase.split()},
        key=len,
        reverse=True,
    )

    if not words:
        return safe_text.replace("\n", "<br>")

    pattern = re.compile(
        r"\b(" + "|".join(re.escape(word) for word in words) + r")\b",
        flags=re.IGNORECASE,
    )
    highlighted = pattern.sub(
        r"<mark style='background-color:#ffb3b3;color:#18201e'>\1</mark>",
        safe_text,
    )
    return highlighted.replace("\n", "<br>")


@st.cache_data(show_spinner=False)
def load_language_dataset(language):
    if language == "Bahasa Indonesia":
        paths = [
            DATA_DIR / "final_preprocessed_indonesia_spam_dataset.csv",
        ]
        frames = [pd.read_csv(path) for path in paths if path.exists()]
        if not frames:
            raise FileNotFoundError("Indonesian dataset file was not found.")
        dataset = pd.concat(frames, ignore_index=True)
    else:
        path = DATA_DIR / "final_preprocessed_emails_dataset.csv"
        if not path.exists():
            raise FileNotFoundError("English dataset file was not found.")
        dataset = pd.read_csv(path)

    if "raw_text" not in dataset.columns:
        if "Pesan" in dataset.columns:
            dataset["raw_text"] = dataset["Pesan"]
        elif "text" in dataset.columns:
            dataset["raw_text"] = dataset["text"]
    if "cleaned_text" not in dataset.columns and "text" in dataset.columns:
        dataset["cleaned_text"] = dataset["text"]
    if "label_num" not in dataset.columns and "label" in dataset.columns:
        dataset["label_num"] = (
            dataset["label"]
            .astype(str)
            .str.lower()
            .map({"ham": 0, "legitimate": 0, "spam": 1, "phishing": 1})
        )

    dataset["label_num"] = pd.to_numeric(dataset["label_num"], errors="coerce")
    dataset = dataset.dropna(subset=["raw_text", "label_num"]).copy()
    dataset["label_num"] = dataset["label_num"].astype(int)
    dataset["display_label"] = dataset["label_num"].map(
        {0: "Legitimate", 1: "Spam / Phishing"}
    )
    if "word_count" not in dataset.columns:
        dataset["word_count"] = dataset["raw_text"].astype(str).str.split().str.len()
    return dataset


@st.cache_data(show_spinner=False)
def load_model_comparison(language):
    filename = (
        "model_comparison_indonesia.csv"
        if language == "Bahasa Indonesia"
        else "model_comparison.csv"
    )
    comparison = pd.read_csv(MODEL_DIR / filename)
    return comparison


@st.cache_data(show_spinner=False)
def load_bert_confusion_matrix(language):
    if language == "Bahasa Indonesia":
        matrix_path = MODEL_DIR / "model_confusion_matrices_indonesia.csv"
        target_model = "Indonesian BERT"
        fallback = np.array([[245, 5], [3, 271]])
    else:
        matrix_path = MODEL_DIR / "model_confusion_matrices.csv"
        target_model = "BERT"
        fallback = np.array([[822, 4], [1, 272]])

    if not matrix_path.exists():
        return fallback

    matrix_df = pd.read_csv(matrix_path)
    matrix_df = matrix_df[matrix_df["Model"] == target_model]
    if matrix_df.empty:
        return fallback

    rows = {
        str(row["Actual"]).lower(): [
            int(row["Predicted Ham"]),
            int(row["Predicted Spam"]),
        ]
        for _, row in matrix_df.iterrows()
    }
    ham_row = rows.get("ham")
    spam_row = rows.get("spam")
    if ham_row is None or spam_row is None:
        return fallback
    return np.array([ham_row, spam_row])


def page_heading(kicker, title, description):
    st.markdown(f'<div class="app-kicker">{kicker}</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<div class="app-summary">{description}</div>', unsafe_allow_html=True)


def fixed_choice(label, options, key, default=None):
    if default is None:
        default = options[0]
    if st.session_state.get(key) not in options:
        st.session_state[key] = default

    st.markdown(label)
    with st.popover(
        st.session_state[key],
        use_container_width=True,
    ):
        st.radio(
            label,
            options,
            key=key,
            label_visibility="collapsed",
        )
    return st.session_state[key]


def scroll_to_top_on_page_change(page):
    if st.session_state.get("last_page") == page:
        return

    st.session_state["last_page"] = page
    components.html(
        """
        <script>
        function scrollEverywhere() {
            const doc = window.parent.document;
            const targets = [
                window.parent,
                doc.documentElement,
                doc.body,
                doc.querySelector('[data-testid="stAppViewContainer"]'),
                doc.querySelector('[data-testid="stAppViewContainer"] > section'),
                doc.querySelector('section.main'),
                doc.querySelector('.stApp')
            ].filter(Boolean);

            targets.forEach((target) => {
                try {
                    if (target.scrollTo) {
                        target.scrollTo(0, 0);
                        target.scrollTo({top: 0, left: 0, behavior: 'auto'});
                    }
                    target.scrollTop = 0;
                    target.scrollLeft = 0;
                } catch (error) {}
            });
        }

        [0, 50, 150, 300, 600, 1000].forEach((delay) => {
            window.setTimeout(scrollEverywhere, delay);
        });
        </script>
        """,
        height=0,
        width=0,
    )


def render_home(language, active_model_type):
    page_heading(
        "NLP Group 5 Project",
        "Bilingual Email Spam Detector",
        "A practical classifier for identifying legitimate and unwanted email "
        "content in English and Bahasa Indonesia.",
    )
    st.divider()
    problem_col, scope_col = st.columns([1.25, 1])
    with problem_col:
        st.subheader("The problem")
        st.write(
            "Spam and phishing messages waste attention and can expose users "
            "to fraud, credential theft, and malicious links. Their wording "
            "also changes across languages, so one English-only model is not "
            "enough for local email use."
        )
        st.write(
            "This project keeps separate English and Indonesian training pipelines "
            "while presenting them through one consistent application."
        )
    with scope_col:
        st.subheader("Current deployment")
        st.metric("Selected language", language)
        st.metric("Default model", format_model_name(active_model_type))
        st.caption("Each language loads its own model and training artifacts.")

    st.subheader("How to use the app")
    steps = [
        "Choose English or Bahasa Indonesia in the sidebar.",
        "Open Text Analyzer and paste the email subject and body.",
        "Select Analyze Email to see the class, confidence, and word influence.",
        "Use the remaining pages to inspect data, charts, and model details.",
    ]
    for number, step in enumerate(steps, start=1):
        st.markdown(
            f'<p><span class="step-number">{number}</span>{step}</p>',
            unsafe_allow_html=True,
        )

    st.subheader("Team members")
    for member in TEAM_MEMBERS:
        st.write(member)


def render_text_analyzer(
    language,
    active_model_type,
    active_model_path,
    active_vectorizer_path,
):
    text = TRANSLATIONS[language]
    page_heading(
        text["section_label"],
        text["title"],
        text["description"],
    )
    st.info(text["language_notice"])
    st.caption(
        f'{text["model"]}: {format_model_name(active_model_type)}'
    )

    examples = {
        "English": {
            "Legitimate meeting": (
                "Subject: Project meeting\n\nHi team, our weekly project meeting "
                "is tomorrow at 10:00 AM in Seminar Room 2. Please bring your "
                "latest results for discussion."
            ),
            "Suspicious prize": (
                "Subject: You won a cash prize\n\nCongratulations! You have been "
                "selected for an exclusive reward. Click the link immediately "
                "and confirm your bank details to receive the money."
            ),
        },
        "Bahasa Indonesia": {
            "Rapat resmi": (
                "Subjek: Rapat proyek\n\nHalo semua, rapat proyek mingguan "
                "akan diadakan besok pukul 10 pagi di Ruang Seminar 2."
            ),
            "Hadiah mencurigakan": (
                "Subjek: Anda memenangkan hadiah tunai\n\nSelamat! Klik tautan ini "
                "segera dan konfirmasi informasi rekening bank Anda untuk menerima hadiah."
            ),
        },
    }
    input_key = f"analyzer_email_{language}"
    result_key = f"analyzer_result_{language}"

    def update_input(value):
        st.session_state[input_key] = value
        st.session_state.pop(result_key, None)

    preset_col, load_col, clear_col = st.columns([2.3, 1, 1])
    with preset_col:
        preset_name = fixed_choice(
            text["try_example"],
            list(examples[language]),
            key=f"preset_{language}",
        )
    load_col.button(
        text["load_example"],
        on_click=update_input,
        args=(examples[language][preset_name],),
        use_container_width=True,
    )
    clear_col.button(
        text["clear"],
        on_click=update_input,
        args=("",),
        use_container_width=True,
    )

    email_text = st.text_area(
        text["email_content"],
        height=240,
        placeholder=text["placeholder"],
        key=input_key,
    )
    analyze = st.button(
        text["analyze"],
        type="primary",
        use_container_width=True,
    )
    if analyze:
        if not email_text.strip():
            st.warning(text["empty"])
            st.session_state.pop(result_key, None)
        else:
            try:
                with st.spinner(text["loading"]):
                    resources = load_model_resources(
                        active_model_type,
                        active_model_path,
                        active_vectorizer_path,
                    )
                    if active_model_type == "bert":
                        prediction, probabilities, processed_text, impacts = predict_bert(
                            email_text,
                            resources,
                        )
                        explanation_error = None
                        try:
                            with st.spinner(text["explaining"]):
                                impacts = explain_bert_words(email_text, resources)
                        except Exception as error:
                            impacts = []
                            explanation_error = str(error)
                    else:
                        prediction, probabilities, processed_text, impacts = predict_classic(
                            email_text,
                            resources,
                            language,
                        )
                        explanation_error = None
                st.session_state[result_key] = {
                    "prediction": prediction,
                    "probabilities": [float(value) for value in probabilities],
                    "processed_text": processed_text,
                    "impacts": impacts,
                    "explanation_error": explanation_error,
                    "source_text": email_text,
                }
            except Exception as error:
                st.session_state.pop(result_key, None)
                st.error(f'{text["error"]}: {error}')

    result = st.session_state.get(result_key)
    if not result:
        return
    if result["source_text"] != email_text:
        st.info(text["changed"])
        return

    prediction = result["prediction"]
    probabilities = result["probabilities"]
    impacts = result["impacts"]
    explanation_error = result.get("explanation_error")
    confidence = float(probabilities[prediction])
    label = text["spam"] if prediction == 1 else text["ham"]
    st.divider()
    result_col, confidence_col, model_col = st.columns(3)
    result_col.metric(text["prediction"], label)
    confidence_col.metric(text["confidence"], f"{confidence:.2%}")
    model_col.metric(text["model"], format_model_name(active_model_type))

    probability_df = pd.DataFrame(
        {
            text["class"]: [text["ham_chart"], text["spam_chart"]],
            text["probability"]: probabilities,
        }
    )
    st.subheader(text["class_probabilities"])
    probability_chart = (
        alt.Chart(probability_df)
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            x=alt.X(f'{text["probability"]}:Q', scale=alt.Scale(domain=[0, 1])),
            y=alt.Y(f'{text["class"]}:N', sort=None, title=None),
            color=alt.Color(
                f'{text["class"]}:N',
                scale=alt.Scale(range=["#159587", "#e76f51"]),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip(f'{text["class"]}:N'),
                alt.Tooltip(f'{text["probability"]}:Q', format=".2%"),
            ],
        )
        .properties(height=120)
    )
    st.altair_chart(probability_chart, use_container_width=True, theme="streamlit")

    st.subheader(text["explanation_title"])
    if impacts:
        st.markdown(
            highlight_bert_impacts(result["source_text"], impacts),
            unsafe_allow_html=True,
        )
        explanation_df = pd.DataFrame(
            {
                text["word_phrase"]: [item["word"] for item in impacts],
                text["influence"]: [
                    text["spam_support"] if item["impact"] > 0 else text["ham_support"]
                    for item in impacts
                ],
                text["impact_score"]: [round(item["impact"], 5) for item in impacts],
            }
        )
        st.dataframe(explanation_df, use_container_width=True, hide_index=True)
    else:
        if explanation_error:
            st.warning(f'{text["explanation_error"]} ({explanation_error})')
        st.info(text["no_explanation"])

    st.caption(
        text["explanation_note"]
        if active_model_type == "bert"
        else text["classic_note"]
    )
    with st.expander(text["processed"]):
        st.write(result["processed_text"])


def render_data_explorer(language):
    page_heading(
        "Dataset inspection",
        "Data Explorer",
        f"Explore the labeled {language} email data used by the training pipeline.",
    )
    try:
        dataset = load_language_dataset(language)
    except Exception as error:
        st.error(str(error))
        return

    legitimate = int((dataset["label_num"] == 0).sum())
    spam = int((dataset["label_num"] == 1).sum())
    metrics = st.columns(4)
    metrics[0].metric("Total emails", f"{len(dataset):,}")
    metrics[1].metric("Legitimate", f"{legitimate:,}")
    metrics[2].metric("Spam / phishing", f"{spam:,}")
    metrics[3].metric("Average words", f'{dataset["word_count"].mean():.1f}')

    st.subheader("Browse records")
    filter_col, search_col, size_col = st.columns([1, 2, 1])
    with filter_col:
        label_filter = fixed_choice(
            "Label",
            ["All", "Legitimate", "Spam / Phishing"],
            key=f"data_label_filter_{language}",
        )
    search_term = search_col.text_input(
        "Search email text",
        placeholder="Enter a word or phrase...",
    )
    sample_size = size_col.slider("Rows", 5, 50, 20, 5)

    filtered = dataset
    if label_filter != "All":
        filtered = filtered[filtered["display_label"] == label_filter]
    if search_term.strip():
        filtered = filtered[
            filtered["raw_text"].astype(str).str.contains(
                search_term.strip(),
                case=False,
                regex=False,
                na=False,
            )
        ]

    st.caption(f"{len(filtered):,} matching records")
    sample = filtered.head(sample_size)[
        ["raw_text", "display_label", "word_count"]
    ].rename(
        columns={
            "raw_text": "Email text",
            "display_label": "Label",
            "word_count": "Words",
        }
    )
    if sample.empty:
        st.info("No emails match the current filters.")
    else:
        st.dataframe(sample, use_container_width=True, hide_index=True)

    pie_col, bar_col = st.columns(2)
    distribution = dataset["display_label"].value_counts()
    with pie_col:
        st.subheader("Class share")
        pie_data = distribution.rename_axis("Label").reset_index(name="Emails")
        pie_data["Percent"] = pie_data["Emails"] / pie_data["Emails"].sum()
        pie_base = alt.Chart(pie_data)
        pie_chart = (
            pie_base
            .mark_arc(innerRadius=58, outerRadius=105, strokeWidth=2)
            .encode(
                theta=alt.Theta("Emails:Q"),
                color=alt.Color(
                    "Label:N",
                    scale=alt.Scale(
                        domain=["Legitimate", "Spam / Phishing"],
                        range=["#159587", "#e76f51"],
                    ),
                    legend=alt.Legend(title=None, orient="bottom"),
                ),
                tooltip=[
                    alt.Tooltip("Label:N"),
                    alt.Tooltip("Emails:Q", format=","),
                    alt.Tooltip("Percent:Q", format=".1%"),
                ],
            )
            .properties(height=300)
        )
        pie_labels = (
            pie_base
            .mark_text(radius=82, fontSize=14, fontWeight="bold", color="white")
            .encode(
                theta=alt.Theta("Emails:Q"),
                text=alt.Text("Percent:Q", format=".1%"),
            )
        )
        st.altair_chart(
            pie_chart + pie_labels,
            use_container_width=True,
            theme="streamlit",
        )
    with bar_col:
        st.subheader("Label totals")
        total_data = distribution.rename_axis("Label").reset_index(name="Emails")
        total_axis_max = max(int(total_data["Emails"].max() * 1.14), 1)
        total_base = alt.Chart(total_data)
        total_bars = (
            total_base
            .mark_bar(cornerRadiusEnd=4)
            .encode(
                x=alt.X(
                    "Emails:Q",
                    title="Emails",
                    scale=alt.Scale(domain=[0, total_axis_max]),
                ),
                y=alt.Y("Label:N", sort=None, title=None),
                color=alt.Color(
                    "Label:N",
                    scale=alt.Scale(
                        domain=["Legitimate", "Spam / Phishing"],
                        range=["#159587", "#e76f51"],
                    ),
                    legend=None,
                ),
                tooltip=[
                    alt.Tooltip("Label:N"),
                    alt.Tooltip("Emails:Q", format=","),
                ],
            )
        )
        total_labels = (
            total_base
            .mark_text(
                align="right",
                baseline="middle",
                dx=-8,
                fontWeight="bold",
                color="#ffffff",
            )
            .encode(
                x=alt.X(
                    "Emails:Q",
                    scale=alt.Scale(domain=[0, total_axis_max]),
                ),
                y=alt.Y("Label:N", sort=None),
                text=alt.Text("Emails:Q", format=","),
            )
        )
        st.altair_chart(
            (total_bars + total_labels).properties(height=300),
            use_container_width=True,
            theme="streamlit",
        )

    st.subheader("Dataset statistics")
    statistics = pd.DataFrame(
        {
            "Statistic": [
                "Rows",
                "Columns",
                "Duplicate email texts",
                "Minimum words",
                "Median words",
                "Maximum words",
            ],
            "Value": [
                len(dataset),
                len(dataset.columns),
                int(dataset["raw_text"].duplicated().sum()),
                int(dataset["word_count"].min()),
                int(dataset["word_count"].median()),
                int(dataset["word_count"].max()),
            ],
        }
    )
    st.dataframe(statistics, use_container_width=True, hide_index=True)


def render_visualizations(language):
    page_heading(
        "Patterns and evaluation",
        "Visualizations",
        "See the vocabulary, class balance, evaluation matrix, and model comparison at a glance.",
    )
    try:
        dataset = load_language_dataset(language)
        comparison = load_model_comparison(language)
    except Exception as error:
        st.error(str(error))
        return

    word_section = st.container()
    distribution_section = st.container()
    evaluation_section = st.container()

    with word_section:
        st.subheader("Word patterns")
        control_col, count_col = st.columns([1.3, 1])
        with control_col:
            cloud_label = fixed_choice(
                "Words from",
                ["All emails", "Legitimate", "Spam / Phishing"],
                key=f"word_cloud_filter_{language}",
            )
        max_words = count_col.slider("Number of words", 40, 180, 100, 20)
        cloud_dataset = dataset
        if cloud_label != "All emails":
            cloud_dataset = dataset[dataset["display_label"] == cloud_label]
        text_column = "cleaned_text" if "cleaned_text" in dataset.columns else "raw_text"
        corpus = " ".join(cloud_dataset[text_column].dropna().astype(str).tolist())
        try:
            from wordcloud import WordCloud

            word_colors = ["#159587", "#d65f45", "#3978c5", "#8c5bb8", "#8a6d1f"]

            def word_color(word, **_kwargs):
                return word_colors[sum(ord(character) for character in word) % len(word_colors)]

            cloud = WordCloud(
                width=1400,
                height=460,
                background_color=None,
                mode="RGBA",
                color_func=word_color,
                max_words=max_words,
                collocations=False,
            ).generate(corpus)
            st.image(
                cloud.to_array(),
                caption=f"Most frequent terms in {cloud_label.lower()}",
                use_container_width=True,
            )
        except ImportError:
            st.warning("Install the wordcloud package to display this chart.")

    with distribution_section:
        st.divider()
        st.subheader("Dataset distribution")
        label_col, length_col = st.columns(2)
        with label_col:
            st.subheader("Label distribution")
            distribution = (
                dataset["display_label"]
                .value_counts()
                .rename_axis("Label")
                .reset_index(name="Emails")
            )
            label_chart = (
                alt.Chart(distribution)
                .mark_bar(cornerRadiusEnd=4)
                .encode(
                    x=alt.X("Emails:Q"),
                    y=alt.Y("Label:N", sort=None, title=None),
                    color=alt.Color(
                        "Label:N",
                        scale=alt.Scale(range=["#159587", "#e76f51"]),
                        legend=None,
                    ),
                    tooltip=["Label:N", "Emails:Q"],
                )
                .properties(height=260)
            )
            st.altair_chart(label_chart, use_container_width=True, theme="streamlit")
        with length_col:
            st.subheader("Email length distribution")
            length_data = dataset[["display_label", "word_count"]].copy()
            length_data["word_count"] = length_data["word_count"].clip(upper=500)
            length_chart = (
                alt.Chart(length_data)
                .mark_bar(opacity=0.66)
                .encode(
                    x=alt.X(
                        "word_count:Q",
                        bin=alt.Bin(maxbins=35),
                        title="Word count (clipped at 500)",
                    ),
                    y=alt.Y("count():Q", title="Emails"),
                    color=alt.Color(
                        "display_label:N",
                        title=None,
                        scale=alt.Scale(range=["#159587", "#e76f51"]),
                    ),
                    tooltip=[alt.Tooltip("count():Q", title="Emails")],
                )
                .properties(height=260)
            )
            st.altair_chart(length_chart, use_container_width=True, theme="streamlit")

    with evaluation_section:
        st.divider()
        st.subheader("Model evaluation")
        matrix_col, comparison_col = st.columns([1, 1.35])
        with matrix_col:
            st.subheader("BERT confusion matrix")
            matrix = load_bert_confusion_matrix(language)
            maximum = max(int(matrix.max()), 1)

            def matrix_cell(value):
                opacity = 0.10 + (0.72 * int(value) / maximum)
                return (
                    '<div class="confusion-cell" '
                    f'style="background:rgba(21,149,135,{opacity:.3f})">'
                    f"{int(value):,}</div>"
                )

            matrix_html = (
                '<div class="confusion-grid">'
                '<div></div>'
                '<div class="confusion-label">Predicted legitimate</div>'
                '<div class="confusion-label">Predicted spam</div>'
                '<div class="confusion-label">Actual legitimate</div>'
                f"{matrix_cell(matrix[0, 0])}{matrix_cell(matrix[0, 1])}"
                '<div class="confusion-label">Actual spam</div>'
                f"{matrix_cell(matrix[1, 0])}{matrix_cell(matrix[1, 1])}"
                "</div>"
            )
            st.markdown(matrix_html, unsafe_allow_html=True)
            st.caption("Held-out evaluation results recorded during notebook training.")
        with comparison_col:
            st.subheader("Model comparison")
            available_metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
            highlighted_metric = st.radio(
                "Highlight metric",
                ["Show all", *available_metrics],
                horizontal=True,
            )
            metric_colors = ["#1677c8", "#ef4444", "#f59e0b", "#159587"]
            if highlighted_metric == "Show all":
                series_colors = metric_colors
            else:
                series_colors = [
                    color if metric == highlighted_metric else "#4b5563"
                    for metric, color in zip(available_metrics, metric_colors)
                ]
            chart_data = comparison.set_index("Model")[available_metrics]
            st.bar_chart(
                chart_data,
                y=available_metrics,
                y_label="Score (0 to 1)",
                color=series_colors,
                stack=False,
                height=340,
            )
            if highlighted_metric == "Show all":
                st.caption("All four evaluation metrics are shown in their assigned colors.")
            else:
                st.caption(
                    f"{highlighted_metric} remains in color while the other metrics are darkened."
                )
            display_comparison = comparison.copy()
            display_comparison[available_metrics] = display_comparison[
                available_metrics
            ].map(lambda value: f"{value:.2%}")
            st.dataframe(
                display_comparison,
                use_container_width=True,
                hide_index=True,
            )


def render_model_info(language, active_model_type):
    page_heading(
        "Training and performance",
        "Model Info",
        "Understand the three classifiers, their measured performance, and how each language pipeline was trained.",
    )
    st.subheader("Models used")
    model_cols = st.columns(3)
    with model_cols[0]:
        st.markdown("#### Multinomial Naive Bayes")
        st.write("A fast probabilistic baseline trained on TF-IDF email features.")
    with model_cols[1]:
        st.markdown("#### Logistic Regression")
        st.write("A linear TF-IDF classifier with interpretable feature weights.")
    with model_cols[2]:
        st.markdown("#### BERT")
        st.write("The default contextual language model, fine-tuned directly on natural email text.")

    st.info(f"Currently deployed for {language}: {format_model_name(active_model_type)}")
    comparison = load_model_comparison(language)
    st.subheader("Performance metrics")
    formatted = comparison.copy()
    metric_columns = ["Accuracy", "Precision", "Recall", "F1-Score"]
    formatted[metric_columns] = formatted[metric_columns].map(lambda value: f"{value:.2%}")
    st.dataframe(formatted, use_container_width=True, hide_index=True)

    best_row = comparison.loc[comparison["F1-Score"].idxmax()]
    metric_cols = st.columns(4)
    metric_cols[0].metric("Best model", best_row["Model"])
    metric_cols[1].metric("Accuracy", f'{best_row["Accuracy"]:.2%}')
    metric_cols[2].metric("Recall", f'{best_row["Recall"]:.2%}')
    metric_cols[3].metric("F1-score", f'{best_row["F1-Score"]:.2%}')

    st.subheader("Training details")
    if language == "Bahasa Indonesia":
        training_rows = [
            ("BERT checkpoint", "bert-base-multilingual-cased"),
            ("Source records", "2,620"),
            ("Split strategy", "80% training / 20% testing, stratified"),
            ("Held-out records", "524"),
            ("BERT input", "Raw natural email text"),
        ]
    else:
        training_rows = [
            ("BERT checkpoint", "bert-base-uncased"),
            ("Source records", "5,728 raw / 5,495 after preprocessing"),
            ("Split strategy", "80% training / 20% testing, stratified"),
            ("BERT input", "Raw natural email text"),
        ]
    training_rows.extend(
        [
            ("Learning rate", "2e-5"),
            ("Epochs", "3"),
            ("Training batch size", "8"),
            ("Evaluation batch size", "16"),
            ("Maximum token length", "256"),
            ("Traditional model features", "TF-IDF preprocessed text"),
        ]
    )
    st.dataframe(
        pd.DataFrame(training_rows, columns=["Setting", "Value"]),
        use_container_width=True,
        hide_index=True,
    )
    with st.expander("Preprocessing summary"):
        if language == "Bahasa Indonesia":
            st.write(
                "HTML is removed, text is lowercased, URLs/emails/numbers are "
                "replaced with semantic placeholders, punctuation is removed, "
                "and whitespace is normalized."
            )
        else:
            st.write(
                "HTML is removed, text is lowercased, URLs/emails/numbers and "
                "punctuation are removed, English stop words are filtered, and "
                "tokens are POS-aware lemmatized."
            )
        st.caption("BERT receives raw text; preprocessing is used for TF-IDF models.")


st.sidebar.markdown("## Email Spam Detector")
st.sidebar.caption("English and Bahasa Indonesia")
page = st.sidebar.radio("Navigate", PAGES)
scroll_to_top_on_page_change(page)
st.sidebar.divider()
language_options = ["English", "Bahasa Indonesia"]
if st.session_state.get("model_language") not in language_options:
    st.session_state["model_language"] = "English"
st.sidebar.markdown("Dataset and model language")
with st.sidebar.popover(
    st.session_state["model_language"],
    icon=":material/language:",
    use_container_width=True,
):
    language = st.radio(
        "Choose language",
        language_options,
        key="model_language",
        label_visibility="collapsed",
    )
active_model_type, active_model_path, active_vectorizer_path = get_language_model_config(language)
st.sidebar.caption(f"Active model: {format_model_name(active_model_type)}")

if page == "Home / About":
    render_home(language, active_model_type)
elif page == "Text Analyzer":
    render_text_analyzer(
        language,
        active_model_type,
        active_model_path,
        active_vectorizer_path,
    )
elif page == "Data Explorer":
    render_data_explorer(language)
elif page == "Visualizations":
    render_visualizations(language)
else:
    render_model_info(language, active_model_type)
