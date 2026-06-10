# Email Spam Detector

This Streamlit app loads one selected spam-classification model and predicts
whether pasted email text is ham or spam.

## Model files

Copy these files from Google Colab into `models/`:

- `tfidf_vectorizer.joblib`
- `logistic_regression_model.joblib`
- `naive_bayes_model.joblib`
- `model_comparison.csv`
- `bert_spam_model/` if BERT is deployed

Select the deployed model in `app.py`:

```python
MODEL_TYPE = "logistic_regression"
```

Valid values are `logistic_regression`, `naive_bayes`, and `bert`.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Important

The preprocessing in `preprocessing.py` must remain identical to the
preprocessing used before training the TF-IDF models. BERT receives raw,
natural email text instead.
