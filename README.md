Student ID: s4043058  (Mohtasem Jahan Siddique)
COSC2815 Assignment 3 - Milestone II: Web-based Data Application (Flask)

HOW TO RUN
  1. python -m venv venv && source venv/bin/activate      (Windows: venv\Scripts\activate)
  2. pip install -r requirements.txt
  3. python app.py
  4. Open http://127.0.0.1:5000 in a browser.
No internet access, database or extra downloads are needed. Everything required is in this folder
(total size is well under 50MB, so no OneDrive link is required).

FILES
  app.py                 Flask application (routes, search, model use, saving reviews)
  train_model.py         Script that trained model/recommender.joblib (already run; re-run is optional)
  model/recommender.joblib   Classification model: bag-of-words (Milestone I regex + stopword list) + logistic regression
  data/assignment3_II.csv    Provided revised dataset (items and reviews)
  data/stopwords_en.txt      Stopword list from Milestone I
  data/new_reviews.json      Created automatically when a review is published on the website
  templates/, static/        HTML templates and CSS
  s4043058_demo.mp4          Video demonstration (3.5 minutes max)

WEBSITE FEATURES
  * Home page: departments, top picks, most reviewed, latest reviews posted by shoppers.
  * Categories: /category/<department> with class filter, sorting and pagination.
  * Item page: description, average rating, % recommended, rating breakdown, reviews, similar items.
  * Search: keyword stemming so "dress"/"dresses" (and top/tops, jeans/jean...) give identical results.
    Shows "N clothing items have matched ..." and item previews, ranked by weighted match
    (title and category words count more than description words, rarer words count more).
    Multi-word queries match all words; if nothing matches all, items matching any word are shown.
  * New review (2 steps, no login needed):
      1. Buyer enters rating, title, review text (and optional age).
      2. The model suggests Recommended / Not recommended with a confidence; the buyer can override
         it, then publishes. The review is saved to data/new_reviews.json, appears on the item page
         and has its own URL (/review/<id>). It remains after restarting the server.

MODEL
  Milestone I best representation: bag-of-words count vectors, logistic regression on review title +
  description (class_weight='balanced'). Held-out accuracy 0.876, macro-F1 0.819.
  The model is loaded from model/recommender.joblib at start-up.

NOTES
  * Item images are generated placeholders (SVG) drawn by the app; no image files are needed.
  * Item titles/descriptions come from the provided "Clothes Title" / "Clothes Description" columns
    (the most common value is used where an item has more than one).
