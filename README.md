# NLP Topic Modeling

An automated topic discovery project using Latent Dirichlet Allocation (LDA) to identify and analyze underlying topics in news headlines from the ABC News dataset.

## Project Overview

This project implements a complete machine learning pipeline for topic modeling that:

- Loads and preprocesses the ABC News dataset
- Cleans text data by removing stopwords and punctuation
- Extracts features using CountVectorizer
- Trains an LDA model with 10 distinct topics
- Provides functionality to classify new text into discovered topics

## Dataset

**Source:** ABC News Article Headlines (abcnews-date-text.csv)

The dataset contains news headlines with associated dates. The project analyzes the textual content to discover underlying topics across multiple articles.

## Project Structure

```
.
├── README.md                      # Project documentation
├── app.py                         # Flask application
├── requirements.txt               # Project dependencies
├── topic_model.ipynb             # Main notebook with full pipeline
├── dataset/
│   └── abcnews-date-text.csv     # ABC News dataset
├── lda_model.pkl                 # Trained LDA model
└── vectorizer.pkl                # Trained CountVectorizer
```

## Key Features

### Data Preprocessing

- Removes punctuation and special characters
- Converts text to lowercase
- Filters out English stopwords
- Tokenization and cleaning

### Feature Extraction

- Uses CountVectorizer for TF (Term Frequency) representation
- Maximum document frequency: 9.5%
- Minimum document frequency: 5
- Removes English stopwords during vectorization

### Model Training

- **Algorithm:** Latent Dirichlet Allocation (LDA)
- **Number of Topics:** 10
- **Learning Method:** Online learning
- **Random State:** 42 (for reproducibility)

### Topic Detection

- Classifies new text into one of the 10 topics
- Displays top 10 words representing the detected topic
- Provides topic name based on top 3 words

## Installation

1. Clone the repository and navigate to the project directory:

```bash
cd "e:\Project\Course Project\NLP Topic Modeling"
```

2. Create a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\Activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Running the Notebook

Open and run `topic_model.ipynb` in Jupyter Notebook:

```bash
jupyter notebook topic_model.ipynb
```

The notebook will:

1. Import required libraries
2. Download NLTK stopwords
3. Load the ABC News dataset
4. Preprocess the data
5. Train the LDA model
6. Save the trained model and vectorizer

### Using the Topic Detection Function

The `detect_new_text_topic()` function classifies new text and displays relevant topics:

```python
new_text = "Your news headline or article text here"
detect_new_text_topic(new_text, vectorizer, lda)
```

**Output Example:**

```
Best topic for the new text: politics-government-parliament

List of words for topic 'politics-government-parliament':
- politics
- government
- parliament
- political
...
```

## Dependencies

Key libraries used:

- **nltk:** Natural Language Toolkit for stopwords and text processing
- **pandas:** Data manipulation and analysis
- **scikit-learn:** Machine learning (CountVectorizer, LDA)
- **joblib:** Model serialization and persistence
- **re:** Regular expressions for text cleaning

See `requirements.txt` for complete list of dependencies.

## Model Output

The trained model generates:

- **lda_model.pkl:** Serialized LDA model with 10 learned topics
- **vectorizer.pkl:** Fitted CountVectorizer for consistent text transformation

## Future Enhancements

- Add interactive visualization of topic distributions
- Implement topic evolution over time using date features
- Create a web interface using Flask (app.py)
- Add more evaluation metrics (coherence score, perplexity)
- Support for dynamic topic count optimization
- Add visualization of word clouds for each topic

## Requirements

- Python 3.7+
- See requirements.txt for package specifications

## Author

Created as a course project for NLP and topic modeling analysis.
