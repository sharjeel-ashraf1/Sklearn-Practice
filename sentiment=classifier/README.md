# Sentiment Classifier

Binary sentiment classification (positive/negative) on the IMDB 50K movie reviews dataset.

## Dataset
- 50,000 IMDB movie reviews, balanced 25k positive / 25k negative
- Source: [IMDB-sentiment-analysis](https://github.com/Ankit152/IMDB-sentiment-analysis) (`IMDB-Dataset.csv`)

## What's new vs. previous projects
- Text vectorization with `TfidfVectorizer`
- Text cleaning (lowercasing, removing HTML tags/punctuation)
- `Pipeline` to combine vectorizer + model
- `MultinomialNB` vs `LogisticRegression` comparison
- `GridSearchCV` for hyperparameter tuning

## Approach
1. Load and clean review text (strip `<br />` tags, punctuation, lowercase)
2. Train/test split (80/20, stratified)
3. Vectorize with TF-IDF
4. Train and compare `MultinomialNB` vs `LogisticRegression`
5. Wrap vectorizer + best model into a `Pipeline`
6. Tune with `GridSearchCV` over `max_features`, `ngram_range`, and `C`
7. Evaluate with accuracy, confusion matrix, and inspect top positive/negative weighted words

## Results
| Model | Test Accuracy |
|---|---|
| MultinomialNB | 85.3% |
| LogisticRegression | 89.2% |
| Tuned Pipeline (GridSearchCV) | 89.9% |

Best params: `{'clf__C': 10, 'tfidf__max_features': 10000, 'tfidf__ngram_range': (1, 2)}`
