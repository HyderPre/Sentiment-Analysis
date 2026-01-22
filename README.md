# Sentiment Analysis with Hinglish Support

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![NLTK](https://img.shields.io/badge/NLTK-3.9-green.svg)](https://www.nltk.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Sentiment analysis system for English text with bilingual support for Hindi/Hinglish using a translate-then-classify approach. Implements ensemble machine learning with 7 different classifiers achieving 54-59% accuracy.

## 🎯 Overview

This project performs sentiment analysis (positive/negative) on tweets and text using an ensemble of machine learning classifiers. The unique feature is **bilingual capability** - it can analyze both English and Hindi/Hinglish text through automatic translation.

### Key Features

- **7 Machine Learning Classifiers**: Naive Bayes, Multinomial NB, Bernoulli NB, Logistic Regression, SGD, SVM, MaxEnt
- **Ensemble Voting**: Hybrid model using majority voting for robust predictions
- **Bilingual Support**: Automatic translation from Hindi/Hinglish to English using Google Translate
- **Fast Loading**: Pre-trained models saved in pickle format (~5 seconds to load vs 5+ minutes training)
- **Comprehensive Preprocessing**: Emoji handling, spell correction, lemmatization, negation handling

## 🏗️ Architecture

### Translate-then-Classify Approach

```
Hindi/Hinglish Input → Google Translate → English Text → Ensemble Classifiers → Sentiment
```

**Why this approach?**
- Limited labeled sentiment data for Hinglish
- NLTK Twitter Sentiment Dataset provides quality English training data
- Translation-based methods are proven effective for cross-lingual NLP
- Enables bilingual capability without needing separate models

## 📊 Dataset

**Training Data**: [NLTK Twitter Samples](https://www.nltk.org/howto/twitter.html)
- Labeled positive and negative tweets
- 40,000 tweets used for training
- 85/15 train-test split

**Note**: Models are trained on English tweets only. Hindi/Hinglish support is achieved through translation preprocessing.

## 🔧 Installation

### Prerequisites

- Python 3.13+
- Virtual environment (recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/HyderPre/Sentiment-Analysis.git
cd Sentiment-Analysis

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install nltk scikit-learn pandas deep-translator textblob pyspellchecker emoji beautifulsoup4
```

### Download NLTK Data

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('vader_lexicon')
nltk.download('twitter_samples')
```

## 🚀 How to Run

### Step 1: Clone the Repository

```bash
git clone https://github.com/HyderPre/Sentiment-Analysis.git
cd Sentiment-Analysis
```

### Step 2: Set Up Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate it
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download NLTK Data

Run this once to download required NLTK datasets:

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('vader_lexicon')
nltk.download('twitter_samples')
```

### Step 5: Open Jupyter Notebook

```bash
jupyter notebook "NLP Project.ipynb"
```

### Step 6: Load Models & Run

In the notebook, run these cells in order:

1. **Cell 9**: Import libraries (~20 seconds)
2. **Cell 12**: Download NLTK data (~1 second)
3. **Cell 14**: Load pre-trained models (~0.5 seconds) ✅ **Models loaded!**
4. **Cells 18-24**: Load helper functions (preprocessing)
5. **Cell 26**: list_to_dict function
6. **Cell 27**: hybrid function
7. **Cell 50**: EnsembleClassifier
8. **Cell 55**: features function
9. **Cell 56**: text_classify function ✅ **Ready for English!**
10. **Cells 59-61**: hinglish functions ✅ **Ready for Hinglish!**

**Total time: ~25 seconds** (vs 5+ minutes if training from scratch!)

### Step 7: Test Predictions

```python
# For English text
text_classify("I love this amazing product!")

# For Hindi/Hinglish text
func("मैं बहुत खुश हूं")  # "I am very happy"
func("Yeh movie bahut acchi hai")  # Mixed Hinglish
```

### Example Outputs

```python
>>> text_classify("This is awesome!")
['awesome']
Tweet given by user: This is awesome!
naive bayes classifier
This Tweet is  positive
------------------------------
Multinomail naive bayes classifier
This Tweet is  positive
------------------------------
...
Hybrid model
This Tweet is  positive
```

## 📝 Preprocessing Pipeline

1. **Emoji Handling**: Convert emojis to text descriptions
2. **Noise Removal**: Remove URLs, handles, special characters
3. **Text Normalization**: 
   - Expand contractions ("can't" → "cannot")
   - Convert to lowercase
   - Remove stopwords (except 'not')
4. **Spell Correction**: Fix misspellings using PySpellChecker
5. **Lemmatization**: Reduce words to base form with POS tagging
6. **Negation Handling**: Replace words after 'not' with antonyms
7. **Feature Extraction**: Extract adjectives using word features list

## 🤖 Models & Performance

| Classifier | Accuracy |
|-----------|----------|
| Naive Bayes | 54.00% |
| Multinomial NB | 59.33% |
| Bernoulli NB | 54.67% |
| Logistic Regression | 58.67% |
| SGD Classifier | 56.00% |
| SVC | 57.33% |
| MaxEnt | 55.33% |
| **Hybrid (Ensemble)** | **Majority Vote** |

## 🔍 Project Structure

```
Sentiment-Analysis/
├── NLP Project.ipynb          # Main notebook with all code
├── sentiment_models.pkl        # Pre-trained model weights
├── english-adjectives.txt      # Feature word list
├── cleaned_tweet.txt          # Processed training data
├── test_data.csv              # Test dataset
├── .gitignore                 # Git ignore rules
├── README.md                  # This file
├── Presentation/              # Project presentation
├── Report/                    # Project documentation
└── Screenshots/               # Result screenshots
```

## 📚 Dependencies

- **nltk** (3.9+): NLP toolkit for tokenization, lemmatization, POS tagging
- **scikit-learn**: Machine learning classifiers
- **deep-translator** (1.11.4): Google Translate API for Hindi/Hinglish
- **pyspellchecker**: Spell correction
- **textblob**: (imported but not actively used)
- **pandas**: Data manipulation
- **emoji**: Emoji processing
- **beautifulsoup4**: HTML cleaning

## 🎓 How It Works

### For English Text:
1. Preprocess text (clean, spell check, lemmatize)
2. Extract adjective features
3. Pass to all 7 classifiers
4. Return individual predictions + ensemble vote

### For Hindi/Hinglish Text:
1. Detect language (auto-detection via Google Translate)
2. Translate to English
3. Follow English text pipeline (steps 1-4 above)

## ⚠️ Limitations

- **Translation dependency**: Accuracy depends on Google Translate quality
- **Cultural context**: Sentiment nuances may be lost in translation
- **Code-mixing**: Complex Hinglish slang may not translate accurately
- **Training data**: Models trained only on English tweets
- **Accuracy**: 54-59% accuracy range (typical for Twitter sentiment analysis)

## 🔮 Future Improvements

- [ ] Train on native Hinglish labeled dataset when available
- [ ] Implement deep learning models (LSTM, BERT)
- [ ] Add neutral sentiment category
- [ ] Support for more Indian languages
- [ ] Real-time Twitter API integration
- [ ] Web interface for easy testing

## 📄 License

This project is licensed under the MIT License.

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 🙏 Acknowledgments

- NLTK team for the Twitter Samples dataset
- Google Translate API for translation services
- Scikit-learn for machine learning tools

## 📧 Contact

**Hyder** - [@HyderPre](https://github.com/HyderPre)

Project Link: [https://github.com/HyderPre/Sentiment-Analysis](https://github.com/HyderPre/Sentiment-Analysis)

---

⭐ Star this repo if you found it helpful!
