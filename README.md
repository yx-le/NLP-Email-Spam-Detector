# Bilingual Email Spam Detector

This Streamlit app classifies English and Bahasa Indonesia email text as
legitimate or spam/phishing. Each language uses its own trained model files,
with BERT selected by default.

## App sections

- Home / About: project purpose, instructions, and team information
- Text Analyzer: prediction, confidence, probabilities, and word influence
- Data Explorer: samples, dataset statistics, and class distribution
- Visualizations: word cloud, label and length charts, confusion matrix, and
  model comparison
- Model Info: model explanations, performance metrics, and training settings

## Model files

The `models/` directory contains separate English and Indonesian artifacts:

- `tfidf_vectorizer.joblib`
- `logistic_regression_model.joblib`
- `naive_bayes_model.joblib`
- `model_comparison.csv`
- `bert_spam_model/`
- `model_comparison_indonesia.csv`
- `bert_indonesia_spam_model/`

The English default model can be overridden through an environment variable:

```powershell
$env:SPAM_MODEL_TYPE = "bert"
```

Valid English values are `logistic_regression`, `naive_bayes`, and `bert`.
Bahasa Indonesia uses the Group 5 `bert_indonesia_spam_model/` checkpoint.

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Important

The preprocessing in `preprocessing.py` must remain identical to the
preprocessing used before training the TF-IDF models. BERT receives raw,
natural email text instead.
