import html
import os
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from preprocessing import preprocess_text


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

# Change this after comparing your models:
# "logistic_regression", "naive_bayes", or "bert"
MODEL_TYPE = os.getenv("SPAM_MODEL_TYPE", "bert")

MODEL_FILES = {
    "logistic_regression": MODEL_DIR / "logistic_regression_model.joblib",
    "naive_bayes": MODEL_DIR / "naive_bayes_model.joblib",
    "bert": MODEL_DIR / "bert_spam_model",
}


st.set_page_config(
    page_title="Email Spam Detector",
    page_icon="E",
    layout="wide",
)

TRANSLATIONS = {
    "English": {
        "title": "Email Spam Detector",
        "description": (
            "Paste an English email below to classify it as legitimate "
            "(ham) or spam."
        ),
        "model": "Deployed model",
        "language_notice": (
            "The interface is multilingual, but the current BERT model was "
            "trained on English emails. Chinese and Malay email predictions "
            "may be unreliable."
        ),
        "email_content": "Email content",
        "placeholder": "Paste the email subject and message here...",
        "analyze": "Analyze Email",
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
        "bert_explanation": (
            "BERT does not provide reliable indicator words from simple "
            "coefficients. An explanation method such as SHAP can be added."
        ),
        "processed": "Text used by the model",
        "error": "Could not run the model",
        "performance": "Model Performance",
    },
    "Bahasa Melayu": {
        "title": "Pengesan E-mel Spam",
        "description": (
            "Tampal e-mel bahasa Inggeris untuk mengklasifikasikannya "
            "sebagai sah (ham) atau spam."
        ),
        "model": "Model digunakan",
        "language_notice": (
            "Antara muka ini menyokong pelbagai bahasa, tetapi model BERT "
            "semasa dilatih menggunakan e-mel bahasa Inggeris. Ramalan untuk "
            "e-mel bahasa Melayu dan Cina mungkin kurang tepat."
        ),
        "email_content": "Kandungan e-mel",
        "placeholder": "Tampal subjek dan mesej e-mel di sini...",
        "analyze": "Analisis E-mel",
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
        "bert_explanation": (
            "BERT tidak memberikan perkataan petunjuk yang boleh dipercayai "
            "melalui pekali mudah. Kaedah seperti SHAP boleh ditambah."
        ),
        "processed": "Teks yang digunakan oleh model",
        "error": "Model tidak dapat dijalankan",
        "performance": "Prestasi Model",
    },
    "中文": {
        "title": "电子邮件垃圾信息检测器",
        "description": "粘贴英文电子邮件，以检测它是正常邮件还是垃圾邮件。",
        "model": "当前模型",
        "language_notice": (
            "界面支持多种语言，但当前 BERT 模型使用英文邮件训练。"
            "对于中文和马来文邮件，预测结果可能不准确。"
        ),
        "email_content": "电子邮件内容",
        "placeholder": "在此粘贴电子邮件主题和正文...",
        "analyze": "分析电子邮件",
        "empty": "请输入电子邮件内容。",
        "prediction": "预测结果",
        "confidence": "置信度",
        "ham": "正常邮件",
        "spam": "垃圾邮件",
        "class_probabilities": "类别概率",
        "class": "类别",
        "probability": "概率",
        "ham_chart": "正常邮件",
        "spam_chart": "垃圾邮件",
        "indicators": "垃圾邮件提示词",
        "explanation_title": "模型为何作出此预测",
        "spam_support": "支持垃圾邮件",
        "ham_support": "支持正常邮件",
        "explanation_note": (
            "系统通过逐个移除词语并测量垃圾邮件与正常邮件分数的变化来估计词语影响。"
            "这只能解释本次预测，并不表示该词语始终属于垃圾邮件。"
        ),
        "no_explanation": "没有单个词语对本次预测产生明显影响。",
        "bert_explanation": (
            "BERT 无法通过简单的模型系数提供可靠的提示词。"
            "后续可以添加 SHAP 等解释方法。"
        ),
        "processed": "模型使用的文本",
        "error": "无法运行模型",
        "performance": "模型表现",
    },
}


@st.cache_resource
def load_model_resources(model_type):
    if model_type in {"logistic_regression", "naive_bayes"}:
        vectorizer_path = MODEL_DIR / "tfidf_vectorizer.joblib"
        model_path = MODEL_FILES[model_type]

        if not vectorizer_path.exists() or not model_path.exists():
            raise FileNotFoundError(
                "Required Joblib model files were not found in models/."
            )

        return {
            "model": joblib.load(model_path),
            "vectorizer": joblib.load(vectorizer_path),
        }

    if model_type == "bert":
        import torch
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        model_path = MODEL_FILES["bert"]
        if not model_path.exists():
            raise FileNotFoundError(
                "The BERT model directory was not found in models/."
            )

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.eval()

        return {
            "model": model,
            "tokenizer": tokenizer,
            "torch": torch,
        }

    raise ValueError(f"Unsupported model type: {model_type}")


def predict_classic(email_text, resources):
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
    ranked_indices = np.argsort(contributions)[::-1]

    indicators = []
    for position in ranked_indices:
        if contributions[position] <= 0:
            continue

        phrase = feature_names[row.indices[position]]
        if phrase not in indicators:
            indicators.append(phrase)

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
            f"padding:2px 3px;border-radius:3px'>{word}</mark>"
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
        r"<mark style='background-color:#ffb3b3'>\1</mark>",
        safe_text,
    )
    return highlighted.replace("\n", "<br>")


language = st.sidebar.selectbox(
    "Language / Bahasa / 语言",
    ["English", "Bahasa Melayu", "中文"],
)
text = TRANSLATIONS[language]

st.title(text["title"])
st.write(text["description"])
st.info(text["language_notice"])
st.caption(
    f'{text["model"]}: {MODEL_TYPE.replace("_", " ").title()}'
)

email_text = st.text_area(
    text["email_content"],
    height=260,
    placeholder=text["placeholder"],
)

if st.button(text["analyze"], type="primary", use_container_width=True):
    if not email_text.strip():
        st.warning(text["empty"])
        st.stop()

    try:
        resources = load_model_resources(MODEL_TYPE)

        if MODEL_TYPE == "bert":
            prediction, probabilities, processed_text, indicators = (
                predict_bert(email_text, resources)
            )
        else:
            prediction, probabilities, processed_text, indicators = (
                predict_classic(email_text, resources)
            )

        confidence = float(probabilities[prediction])
        label = text["spam"] if prediction == 1 else text["ham"]

        result_col, confidence_col = st.columns(2)
        result_col.metric(text["prediction"], label)
        confidence_col.metric(text["confidence"], f"{confidence:.2%}")

        probability_df = pd.DataFrame(
            {
                text["class"]: [text["ham_chart"], text["spam_chart"]],
                text["probability"]: [
                    float(probabilities[0]),
                    float(probabilities[1]),
                ],
            }
        ).set_index(text["class"])

        st.subheader(text["class_probabilities"])
        st.bar_chart(probability_df)

        if MODEL_TYPE == "bert":
            st.subheader(text["explanation_title"])
            with st.spinner("Generating SHAP explanation..."):
                impacts = explain_bert_words(
                    email_text,
                    resources,
                )

            if impacts:
                st.markdown(
                    highlight_bert_impacts(email_text, impacts),
                    unsafe_allow_html=True,
                )

                explanation_df = pd.DataFrame(
                    {
                        "Word": [item["word"] for item in impacts],
                        "Direction": [
                            (
                                text["spam_support"]
                                if item["impact"] > 0
                                else text["ham_support"]
                            )
                            for item in impacts
                        ],
                        "SHAP impact": [
                            round(item["impact"], 5) for item in impacts
                        ],
                    }
                )
                st.dataframe(explanation_df, use_container_width=True)
            else:
                st.info(text["no_explanation"])

            st.caption(text["explanation_note"])
        elif indicators:
            st.subheader(text["indicators"])
            st.write(", ".join(indicators))
            st.markdown(
                highlight_indicators(email_text, indicators),
                unsafe_allow_html=True,
            )
        else:
            st.subheader(text["indicators"])
            st.write("No strong spam indicators were found in this email.")

        with st.expander(text["processed"]):
            st.write(processed_text)

    except Exception as error:
        st.error(f'{text["error"]}: {error}')


comparison_path = MODEL_DIR / "model_comparison.csv"
if comparison_path.exists():
    st.divider()
    st.subheader(text["performance"])
    comparison_df = pd.read_csv(comparison_path)
    st.dataframe(comparison_df, use_container_width=True)

    if {"Model", "F1-Score"}.issubset(comparison_df.columns):
        st.bar_chart(comparison_df.set_index("Model")[["F1-Score"]])
