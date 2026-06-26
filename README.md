# Bilingual Email Spam Detector - Group 5

An NLP email spam classification app built with BERT and Streamlit. The app
classifies pasted English or Bahasa Indonesia email text as legitimate
(`ham`) or spam/phishing, shows class probabilities, and displays word-impact
explanations when the local SHAP environment supports them.

BERT is selected by default for both languages. English and Bahasa Indonesia
use separate trained model folders and separate dataset files.

## Main features

- English and Bahasa Indonesia spam email classification
- BERT-based prediction with a configurable English model fallback
- Spam/ham prediction result and confidence score
- Class probability chart
- Word influence explanation for BERT predictions
- Data Explorer with sample records, search, dataset statistics, and label
  distribution
- Visualizations page with word clouds, class charts, confusion matrix, and
  model comparison chart
- Model Info page with model descriptions, performance metrics, and training
  settings
- Local Streamlit web application

## App sections

- Home / About: project purpose, instructions, current deployment, and team
  members
- Text Analyzer: email input, prediction, confidence, probabilities, and word
  influence
- Data Explorer: dataset samples, filters, statistics, and class distribution
- Visualizations: word cloud, label distribution, length chart, confusion
  matrix, and model comparison
- Model Info: model explanation, performance metrics, and training details

## Dataset

The project uses separate English and Bahasa Indonesia datasets.

English dataset files:

- `../dataset/emails.csv`
- `../dataset/final_preprocessed_emails_dataset.csv`

The English model uses `emails.csv`, an English email spam dataset. The final
preprocessed dataset contains:

- Raw samples: 5,728
- Final preprocessed samples: 5,495
- Ham emails: 4,129
- Spam emails: 1,366
- Train-test split: 80:20 with stratification

Bahasa Indonesia dataset files:

- `../dataset/email_spam_indo.csv`
- `../dataset/final_preprocessed_indonesia_spam_dataset.csv`

The Bahasa Indonesia mode uses `email_spam_indo.csv`, an Indonesian
email-style spam dataset with:

- `Pesan`: email text
- `Kategori`: label

This dataset is more suitable than an Indonesian SMS dataset because the
project theme is email spam detection.

## Models and results

English model comparison:

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| English BERT | 0.9964 | 0.9927 | 0.9927 | 0.9927 |
| Logistic Regression | 0.9955 | 0.9820 | 1.0000 | 0.9909 |
| Multinomial Naive Bayes | 0.9864 | 1.0000 | 0.9451 | 0.9718 |

Bahasa Indonesia model comparison:

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Indonesian BERT | 0.9847 | 0.9819 | 0.9891 | 0.9855 |

BERT was selected for deployment because it is the strongest deployed model
for the English dataset and the available Group 5 model for Bahasa Indonesia.

## Model files

The app expects model artifacts in `models/`:

```text
models/
|-- bert_spam_model/
|   |-- config.json
|   |-- model.safetensors
|   |-- tokenizer.json
|   |-- tokenizer_config.json
|   `-- training_args.bin
|-- bert_indonesia_spam_model/
|   |-- config.json
|   |-- model.safetensors
|   |-- tokenizer.json
|   |-- tokenizer_config.json
|   `-- training_args.bin
|-- tfidf_vectorizer.joblib
|-- logistic_regression_model.joblib
|-- naive_bayes_model.joblib
|-- model_comparison.csv
|-- model_confusion_matrices.csv
|-- model_comparison_indonesia.csv
`-- model_confusion_matrices_indonesia.csv
```

English mode supports these model types:

- `bert`
- `logistic_regression`
- `naive_bayes`

The English default model can be overridden through an environment variable:

```powershell
$env:SPAM_MODEL_TYPE = "bert"
```

Bahasa Indonesia uses the Group 5 `bert_indonesia_spam_model/` checkpoint.

## Project structure

Current finalized folder layout:

```text
nlp project/
|-- dataset/
|   |-- emails.csv
|   |-- final_preprocessed_emails_dataset.csv
|   |-- email_spam_indo.csv
|   `-- final_preprocessed_indonesia_spam_dataset.csv
|-- Model Trained/
|-- spam_email_detector/
|   |-- app.py
|   |-- preprocessing.py
|   |-- requirements.txt
|   |-- README.md
|   `-- models/
|       |-- model_comparison.csv
|       |-- model_comparison_indonesia.csv
|       |-- model_confusion_matrices.csv
|       |-- model_confusion_matrices_indonesia.csv
|       |-- bert_spam_model/
|       `-- bert_indonesia_spam_model/
`-- nltk_data/
```

For Streamlit Cloud deployments where `app.py`, `dataset/`, and `models/` are
all at the repository root, set the app paths as:

```python
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "dataset"
```

For this local folder, where `app.py` is inside `spam_email_detector/`, the app
uses:

```python
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR.parent / "dataset"
```

## Local installation

Python 3.11 is recommended. Open a terminal in `spam_email_detector`, then run:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL displayed by Streamlit, normally:

```text
http://localhost:8501
```

Stop the app by pressing `Ctrl+C` in the terminal.

## Training notebooks

The Group 5 training notebooks are stored in `Model Trained/`. For the
Indonesian model, run `Model Trained/Indonesia_BERT_Training.ipynb` in Google
Colab and place the exported `bert_indonesia_spam_model/` folder into
`spam_email_detector/models/`.

BERT receives raw natural email text. TF-IDF models use preprocessing from
`preprocessing.py`, so that file must remain consistent with the preprocessing
used during training.

## GitHub note

Do not upload `.venv`, `__pycache__`, or `.pyc` files. They are generated
locally and are not portable.

The BERT `model.safetensors` files are larger than GitHub's normal single-file
upload limit. Use Git LFS, GitHub Releases, cloud storage, or split compressed
archives for model weights.

## Limitations

- English and Bahasa Indonesia are trained as separate models.
- The Bahasa Indonesia dataset appears to be translated/email-style data, so
  real Indonesian email performance may be lower than notebook test
  performance.
- The app is a classifier, not a guaranteed security tool.
- SHAP explanations estimate token influence; they do not prove that a word is
  always spam or ham.
- BERT inference and SHAP explanations can be slow on CPU-only computers.
- If SHAP fails because of package compatibility, the app still shows the model
  prediction and confidence score.

## Libraries

Streamlit, pandas, NumPy, scikit-learn, NLTK, Beautiful Soup, Transformers,
PyTorch, TorchVision, Joblib, SHAP, Matplotlib, Altair, and WordCloud.
