"""Train the recommendation classifier used by the website (Milestone I approach).

Bag-of-words count vectors (Milestone I tokenisation regex + stopword list) on the review
title + description, followed by logistic regression. In Milestone I this was the best
representation. Run once: python train_model.py   ->  model/recommender.joblib
"""
import re
import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

SEED = 42
TOKEN_PATTERN = r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?"          # same regex as Milestone I

df = pd.read_csv('data/assignment3_II.csv')
stopwords = sorted(set(open('data/stopwords_en.txt', encoding='utf-8').read().split()))
text = (df['Title'] + ' ' + df['Review Text']).tolist()   # title + description, concatenated
y = df['Recommended IND'].to_numpy()


def make_model():
    return Pipeline([
        ('bow', CountVectorizer(lowercase=True, token_pattern=TOKEN_PATTERN, stop_words=stopwords, min_df=2)),
        ('clf', LogisticRegression(C=0.1, max_iter=3000, class_weight='balanced', random_state=SEED)),
    ])


# honest estimate on held-out data, then refit on everything for the website
X_tr, X_te, y_tr, y_te = train_test_split(text, y, test_size=0.2, stratify=y, random_state=SEED)
m = make_model().fit(X_tr, y_tr)
pred = m.predict(X_te)
print('held-out accuracy %.3f  macro-F1 %.3f' % (accuracy_score(y_te, pred), f1_score(y_te, pred, average='macro')))

final = make_model().fit(text, y)
joblib.dump(final, 'model/recommender.joblib', compress=3)
print('saved model/recommender.joblib; vocabulary size', len(final.named_steps['bow'].vocabulary_))
for s in ['Love it, so comfortable and flattering!', 'Terrible quality, fell apart and I returned it', 'It is ok']:
    print(round(final.predict_proba([s])[0, 1], 3), s)
