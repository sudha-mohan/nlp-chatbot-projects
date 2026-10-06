# 🏭 Industrial Safety Risk Chatbot (NLP Capstone)

**NLP-based classifier that predicts accident risk level from incident descriptions**

> 🎓 Capstone project for the PG Diploma in AI/ML, Great Lakes Institute of Management / Texas McCombs School of Business.

## 📌 Problem Statement

Employees continue to suffer injuries and fatalities in industrial plants despite safety protocols. Identifying safety risks early from incident descriptions can help prevent serious harm.

This project builds the NLP/ML core of a **chatbot utility** that classifies the accident level from a free-text incident description, to help safety professionals prioritize preventive action.

## 🎯 Project Objective

Design a Machine Learning / Deep Learning model that:

- Accepts an incident description as input
- Predicts the **Accident Level** (I to V, mapped to 0 to 4) and the **Potential Accident Level** (I to VI)
- Is saved to disk so it can be plugged into a chatbot interface

> **Scope note:** this repository covers data analysis, NLP preprocessing, model comparison and model saving. A chat user interface is not included.

## 📊 Dataset

- **Source:** IHMStefanini Industrial Safety and Health Analytics Database (Kaggle)
- **Size:** 425 incident records, 10 columns after cleaning
- **Features:** Date, Country, Local, Industry Sector, Accident Level, Potential Accident Level, Gender, Employee or Third Party, Critical Risk, Description
- **Targets:** Accident Level and Potential Accident Level
- **Link:** [Industrial Safety and Health Analytics Database on Kaggle](https://www.kaggle.com/datasets/ihmstefanini/industrial-safety-and-health-analytics-database)

**Citation:**

```bibtex
@misc{ihmstefanini_2018,
  author       = {IHMStefanini},
  title        = {Industrial Safety and Health Analytics Database},
  year         = {2018},
  publisher    = {Kaggle},
  howpublished = {\url{https://www.kaggle.com/datasets/ihmstefanini/industrial-safety-and-health-analytics-database}}
}
```

## 🔍 Key Findings from EDA

| Insight | Finding |
| --- | --- |
| Most common accident level | Level I |
| Most affected industry | Mining (56.70%) |
| Most affected country | Country_01 (59.33%) |
| Most accidents by month | February (14.59%) |
| Most affected employee type | Third party workers |
| Peak accident year | 2016 (more accidents than 2017) |

## 🧹 NLP Preprocessing

Incident descriptions were cleaned (lowercasing, removing special characters, numbers and punctuation, fixing contractions), lemmatized, and stripped of English stopwords. Text length and word frequency (word clouds) were analyzed before and after cleaning. The maximum description length dropped from 183 words to 102 words.

## 🤖 Models Compared

Text was converted to features with **CountVectorizer** and **TF-IDF** (1-2 grams) and fed to four classifiers: Linear SVC, Random Forest, Gradient Boosting and XGBoost. A deep learning model was also trained: **GloVe (200d) embeddings + Bidirectional LSTM**.

Test accuracy (the classical models use a 15% test split, 64 records; the LSTM uses a 30% split):

| Model | Accident Level | Potential Accident Level |
| --- | --- | --- |
| Linear SVC (CountVectorizer) | 78.1% | 35.9% |
| Linear SVC (TF-IDF) | 78.1% | 37.5% |
| Random Forest (CountVectorizer) | 78.1% | 37.5% |
| Random Forest (TF-IDF) | 78.1% | 37.5% |
| Gradient Boosting (CountVectorizer) | 75.0% | 39.1% |
| Gradient Boosting (TF-IDF) | 75.0% | 40.6% |
| XGBoost (CountVectorizer) | 78.1% | 43.8% |
| XGBoost (TF-IDF) | 76.6% | 40.6% |
| GloVe + BiLSTM | 78.0% | 28.9% |

**Selected model:** Linear SVC with CountVectorizer, which performed well on both targets. It is saved with `pickle` as `accident_level.sav` and `potential_accident.sav`.

### ⚠️ Limitations

- **The data is heavily imbalanced.** In the test split, 50 of 64 records are Level I, so a model that always predicts Level I scores about 78%. The Accident Level accuracies above should not be read as strong predictive performance. The Random Forest classification report confirms it: only Level I is predicted, with zero precision and recall on Levels II to V (macro F1 0.18).
- **Overfitting:** the selected SVC reaches 99.45% train accuracy against 78.12% test accuracy, with 13,415 features on only 361 training records.
- **Potential Accident Level is hard** with this data (about 29% to 44% accuracy).
- With only 425 records, results are sensitive to the split, and no class-balancing technique (resampling or class weights) was applied.
- **Possible next steps:** handle class imbalance, report macro F1 and per-class recall, use cross-validation, and try a transformer-based model.

## 🛠️ Tech Stack

| Category | Tools |
| --- | --- |
| Language | Python 3 |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, WordCloud |
| NLP | NLTK, Contractions, CountVectorizer, TF-IDF, GloVe embeddings |
| Machine Learning | scikit-learn, XGBoost |
| Deep Learning | TensorFlow, Keras (Bidirectional LSTM) |
| Notebook | Jupyter Notebook |

## 📁 Project Structure

```
nlp-chatbot-projects/
│
├── Capstone_NLP_Chatbot.ipynb        # EDA, preprocessing, modelling
├── IHMStefanini_industrial_safety_and_health_database_with_accidents_description.csv
├── README.md
└── requirements.txt
```

## 🚀 How to Run

```bash
# Clone the repository
git clone https://github.com/sudha-mohan/nlp-chatbot-projects.git
cd nlp-chatbot-projects

# Install dependencies
pip install -r requirements.txt

# Open Jupyter Notebook
jupyter notebook
```

**GloVe embeddings (needed only for the BiLSTM section):** download `glove.6B.200d.txt` from the [Stanford GloVe page](https://nlp.stanford.edu/projects/glove/) and place it in the same folder as the notebook. The file is large, so it is not included in this repository.

The notebook loads the CSV by file name, so keep the CSV in the same folder as the notebook. Run the cells from top to bottom.

---

**Sudha Mohan** · Data scientist (ex-TCS) building applied AI and LLM projects, currently the **TNPSC AI Answer Evaluator**, an agentic system built on Gemini function calling.

🔗 [LinkedIn](https://linkedin.com/in/sudha-mohan-696b2a81) · 💻 [GitHub](https://github.com/sudha-mohan)
