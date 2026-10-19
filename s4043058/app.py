"""Milestone II - clothing shopping website (Flask). Student ID: s4043058

Run:  python app.py   then open http://127.0.0.1:5000
"""
import hashlib
import json
import math
import re
import threading
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, Response, abort, redirect, render_template, request, url_for

BASE = Path(__file__).parent
DATA_FILE = BASE / 'data' / 'assignment3_II.csv'
NEW_REVIEWS_FILE = BASE / 'data' / 'new_reviews.json'      # reviews created on the website
MODEL_FILE = BASE / 'model' / 'recommender.joblib'
PER_PAGE = 24
REVIEWS_PER_PAGE = 8

app = Flask(__name__)
write_lock = threading.Lock()

# --------------------------------------------------------------------------- data loading
df = pd.read_csv(DATA_FILE)


def mode(series):
    return series.value_counts().index[0]


# one row per clothing item (Clothing ID)
items = {}
for cid, g in df.groupby('Clothing ID'):
    items[int(cid)] = {
        'id': int(cid),
        'title': mode(g['Clothes Title']),
        'description': mode(g['Clothes Description']),
        'division': mode(g['Division Name']),
        'department': mode(g['Department Name']),
        'class': mode(g['Class Name']),
    }

# all reviews: the original ones (id = row number) followed by the ones created on the website
reviews = []
for rid, r in enumerate(df.itertuples(index=False)):
    reviews.append({'id': rid, 'item': int(r[0]), 'age': int(r[1]), 'title': r[2], 'text': r[3],
                    'rating': int(r[4]), 'recommended': int(r[5]), 'helpful': int(r[6]),
                    'model_label': None, 'new': False})
if NEW_REVIEWS_FILE.exists():
    for r in json.loads(NEW_REVIEWS_FILE.read_text(encoding='utf-8')):
        reviews.append(r)

reviews_by_item = defaultdict(list)


def rebuild_item_stats():
    """Review counts / average rating / share recommended for every item."""
    reviews_by_item.clear()
    for r in reviews:
        reviews_by_item[r['item']].append(r)
    for cid, it in items.items():
        rs = reviews_by_item.get(cid, [])
        it['n_reviews'] = len(rs)
        it['avg_rating'] = round(sum(r['rating'] for r in rs) / len(rs), 1) if rs else 0
        it['pct_rec'] = round(100 * sum(r['recommended'] for r in rs) / len(rs)) if rs else 0


rebuild_item_stats()

# category tree: department -> classes
departments = defaultdict(lambda: {'count': 0, 'classes': Counter()})
for it in items.values():
    departments[it['department']]['count'] += 1
    departments[it['department']]['classes'][it['class']] += 1
DEPARTMENTS = sorted(departments, key=lambda d: -departments[d]['count'])
DEPT_BLURB = {
    'Tops': 'Blouses, knits, sweaters and everyday layers.',
    'Dresses': 'From weekday wraps to occasion dresses.',
    'Bottoms': 'Pants, jeans, skirts and shorts.',
    'Intimate': 'Lounge, swim and soft essentials.',
    'Jackets': 'Outerwear and jackets for every season.',
    'Trend': 'This season\'s standout pieces.',
}

# --------------------------------------------------------------------------- model
model = joblib.load(MODEL_FILE)


def predict_recommendation(title, text):
    """Return (label, probability of 'recommended') for a new review (title + description)."""
    p = float(model.predict_proba([f'{title} {text}'])[0, 1])
    return int(p >= 0.5), p


# --------------------------------------------------------------------------- search
def stem(word):
    """Light suffix stripping so that 'dress' / 'dresses', 'top' / 'tops' ... match."""
    w = word.lower()
    if len(w) > 4 and w.endswith('ies'):
        return w[:-3] + 'y'
    if len(w) > 4 and w.endswith('sses'):
        return w[:-2]
    if len(w) > 4 and re.search(r'(x|z|ch|sh)es$', w):
        return w[:-2]
    if len(w) > 3 and w.endswith('s') and not w.endswith(('ss', 'us', 'is')):
        return w[:-1]
    return w


def tokens(text):
    return [stem(t) for t in re.findall(r"[a-zA-Z]+", text.lower())]


# weighted inverted index: stem -> {item id: weight}. Title/category words count more than description words.
WEIGHTS = {'title': 3.0, 'category': 2.5, 'description': 1.0}
index = defaultdict(lambda: defaultdict(float))
for it in items.values():
    for field, text in (('title', it['title']), ('category', f"{it['class']} {it['department']} {it['division']}"),
                        ('description', it['description'])):
        for t in set(tokens(text)):
            index[t][it['id']] += WEIGHTS[field]


def search_items(query):
    """Items matching ALL keywords (ranked by weighted match); falls back to ANY keyword if none match all."""
    terms = [t for t in dict.fromkeys(tokens(query)) if t]
    if not terms:
        return []
    matched = [set(index.get(t, {})) for t in terms]
    ids = set.intersection(*matched)
    if not ids:
        ids = set.union(*matched)
    n = len(items)
    scores = {}
    for cid in ids:
        s = 0.0
        for t in terms:
            if cid in index.get(t, {}):
                s += index[t][cid] * math.log(1 + n / len(index[t]))      # rarer words count more
        scores[cid] = s + 0.01 * math.log(1 + items[cid]['n_reviews'])
    return [items[c] for c in sorted(ids, key=lambda c: -scores[c])]


# --------------------------------------------------------------------------- placeholder images
PALETTES = [('#ecd9c6', '#c9a27e'), ('#d6e2dc', '#8fb5a3'), ('#e4d3dc', '#b88aa3'), ('#d5dbe8', '#8a9bc4'),
            ('#efe2c4', '#d3b26a'), ('#dfd6cf', '#a8917f'), ('#d4e4ea', '#7fb0c2'), ('#e8d0cb', '#c78d82')]
ICONS = {  # simple garment silhouettes (viewBox 0 0 100 100)
    'dress': 'M38 14 L44 14 L50 24 L56 14 L62 14 L60 34 L70 88 L30 88 L40 34 Z',
    'top': 'M34 18 L44 14 Q50 24 56 14 L66 18 L84 34 L74 46 L66 40 L66 86 L34 86 L34 40 L26 46 L16 34 Z',
    'pants': 'M32 14 L68 14 L72 88 L54 88 L50 40 L46 88 L28 88 Z',
    'skirt': 'M36 16 L64 16 L66 30 L80 88 L20 88 L34 30 Z',
    'jacket': 'M30 16 L44 12 L50 22 L56 12 L70 16 L88 38 L78 52 L70 46 L70 88 L30 88 L30 46 L22 52 L12 38 Z',
}


def icon_for(it):
    c = (it['class'] + ' ' + it['department']).lower()
    for key, name in (('dress', 'dress'), ('skirt', 'skirt'), ('pant', 'pants'), ('jean', 'pants'),
                      ('short', 'pants'), ('legging', 'pants'), ('jacket', 'jacket'), ('outerwear', 'jacket')):
        if key in c:
            return ICONS[name]
    return ICONS['top']


@app.route('/favicon.ico')
def favicon():
    return Response(status=204)


@app.route('/img/<int:item_id>.svg')
def item_image(item_id):
    it = items.get(item_id) or abort(404)
    h = int(hashlib.md5(f"{it['title']}{it['class']}".encode()).hexdigest(), 16)
    c1, c2 = PALETTES[h % len(PALETTES)]
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 500" preserveAspectRatio="xMidYMid slice">'
           f'<defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="{c1}"/>'
           f'<stop offset="1" stop-color="{c2}"/></linearGradient></defs>'
           f'<rect width="400" height="500" fill="url(#g)"/>'
           f'<circle cx="320" cy="90" r="120" fill="#fff" opacity=".12"/>'
           f'<g transform="translate(70 90) scale(2.6)"><path d="{icon_for(it)}" fill="#fff" opacity=".88" '
           f'stroke="#fff" stroke-width="1.5" stroke-linejoin="round"/></g></svg>')
    return Response(svg, mimetype='image/svg+xml', headers={'Cache-Control': 'public, max-age=86400'})


# --------------------------------------------------------------------------- helpers
SORTS = {
    'popular': ('Most reviewed', lambda i: (-i['n_reviews'], i['id'])),
    'rating': ('Highest rated', lambda i: (-i['avg_rating'], -i['n_reviews'])),
    'recommended': ('Most recommended', lambda i: (-i['pct_rec'], -i['n_reviews'])),
    'newest': ('Item number', lambda i: i['id']),
}


def paginate(seq, page, per_page):
    pages = max(1, math.ceil(len(seq) / per_page))
    page = min(max(page, 1), pages)
    return seq[(page - 1) * per_page:page * per_page], page, pages


def page_arg():
    try:
        return int(request.args.get('page', 1))
    except ValueError:
        return 1


@app.template_filter('stars')
def stars(n):
    n = int(round(n))
    return '★' * n + '☆' * (5 - n)


@app.context_processor
def inject_globals():
    def page_url(page):
        """URL of the current page with a different ?page= (keeps the other query parameters)."""
        args = {k: v for k, v in request.args.items() if k != 'page'}
        return url_for(request.endpoint, **(request.view_args or {}), **args, page=page)
    return {'nav_departments': DEPARTMENTS, 'q': request.args.get('q', ''), 'page_url': page_url}


# --------------------------------------------------------------------------- pages
@app.route('/')
def home():
    popular = sorted(items.values(), key=SORTS['popular'][1])
    # featured: well-reviewed items, at most one per product title so the row looks varied
    seen, featured = set(), []
    for it in sorted(items.values(), key=lambda i: (-(i['avg_rating'] >= 4.3), -i['n_reviews'])):
        if it['title'] not in seen and it['n_reviews'] >= 10:
            seen.add(it['title'])
            featured.append(it)
        if len(featured) == 8:
            break
    tiles = []
    for dept in ('Dresses', 'Tops'):                       # the two big tiles use the most reviewed item as the picture
        best = next(i for i in popular if i['department'] == dept)
        tiles.append({'department': dept, 'item': best['id'], 'count': departments[dept]['count']})
    latest = [r for r in reviews if r['new']][::-1][:4]
    stats = {'n_items': len(items), 'n_reviews': len(reviews), 'pct': round(100 * sum(r['recommended'] for r in reviews) / len(reviews))}
    return render_template('home.html', featured=featured, popular=popular[:4], latest=latest, items=items,
                           departments=departments, dept_order=DEPARTMENTS, blurb=DEPT_BLURB, stats=stats, tiles=tiles)


@app.route('/category/<department>')
def category(department):
    if department not in departments:
        abort(404)
    cls = request.args.get('class', '')
    sort = request.args.get('sort', 'popular')
    sort = sort if sort in SORTS else 'popular'
    selected = [i for i in items.values() if i['department'] == department and (not cls or i['class'] == cls)]
    selected.sort(key=SORTS[sort][1])
    page_items, page, pages = paginate(selected, page_arg(), PER_PAGE)
    return render_template('category.html', department=department, classes=departments[department]['classes'].most_common(),
                           cls=cls, sort=sort, sorts=SORTS, items=page_items, total=len(selected), page=page, pages=pages)


@app.route('/search')
def search():
    query = request.args.get('q', '').strip()
    results = search_items(query) if query else []
    page_items, page, pages = paginate(results, page_arg(), PER_PAGE)
    return render_template('search.html', query=query, items=page_items, total=len(results), page=page, pages=pages)


@app.route('/item/<int:item_id>')
def item_page(item_id):
    it = items.get(item_id) or abort(404)
    rs = sorted(reviews_by_item[item_id], key=lambda r: (not r['new'], -r['id'] if r['new'] else -r['helpful'], r['id']))
    star_counts = Counter(r['rating'] for r in rs)
    page_reviews, page, pages = paginate(rs, page_arg(), REVIEWS_PER_PAGE)
    similar = [i for i in sorted(items.values(), key=SORTS['popular'][1])
               if i['class'] == it['class'] and i['id'] != item_id][:4]
    return render_template('item.html', item=it, reviews=page_reviews, total=len(rs), page=page, pages=pages,
                           star_counts=star_counts, similar=similar)


@app.route('/item/<int:item_id>/review/new', methods=['GET', 'POST'])
def new_review(item_id):
    """Step 1: the buyer writes the review. On POST the model suggests a label (step 2: confirm page)."""
    it = items.get(item_id) or abort(404)
    if request.method == 'GET':
        return render_template('new_review.html', item=it, form={}, errors={})
    form = {k: request.form.get(k, '').strip() for k in ('title', 'text', 'rating', 'age')}
    if 'edit' in request.form:                      # 'Edit review' button on the confirm page
        return render_template('new_review.html', item=it, form=form, errors={})
    errors = {}
    if not form['title']:
        errors['title'] = 'Please add a short title.'
    if len(form['text']) < 10:
        errors['text'] = 'Please write at least a sentence (10+ characters).'
    if form['rating'] not in {'1', '2', '3', '4', '5'}:
        errors['rating'] = 'Please choose a star rating.'
    if form['age'] and not (form['age'].isdigit() and 10 <= int(form['age']) <= 110):
        errors['age'] = 'Age must be a number between 10 and 110.'
    if errors:
        return render_template('new_review.html', item=it, form=form, errors=errors), 400
    label, prob = predict_recommendation(form['title'], form['text'])
    return render_template('confirm_review.html', item=it, form=form, label=label, prob=prob)


@app.route('/item/<int:item_id>/review/confirm', methods=['POST'])
def confirm_review(item_id):
    """Step 2: the buyer confirms (or overrides) the suggested label; the review is saved."""
    it = items.get(item_id) or abort(404)
    f = request.form
    title, text = f.get('title', '').strip(), f.get('text', '').strip()
    if not title or len(text) < 10 or f.get('rating') not in {'1', '2', '3', '4', '5'} or f.get('final_label') not in {'0', '1'}:
        abort(400)
    age = int(f['age']) if f.get('age', '').isdigit() else 0
    model_label, _ = predict_recommendation(title, text)       # recomputed on the server, never trusted from the form
    with write_lock:
        review = {'id': len(reviews), 'item': item_id, 'age': age, 'title': title, 'text': text,
                  'rating': int(f['rating']), 'recommended': int(f['final_label']), 'helpful': 0,
                  'model_label': model_label, 'new': True}
        reviews.append(review)
        rebuild_item_stats()
        saved = [r for r in reviews if r['new']]
        tmp = NEW_REVIEWS_FILE.with_suffix('.tmp')
        tmp.write_text(json.dumps(saved, indent=1), encoding='utf-8')
        tmp.replace(NEW_REVIEWS_FILE)
    return redirect(url_for('review_page', review_id=review['id']))


@app.route('/review/<int:review_id>')
def review_page(review_id):
    if review_id < 0 or review_id >= len(reviews):
        abort(404)
    r = reviews[review_id]
    return render_template('review.html', review=r, item=items[r['item']])


@app.errorhandler(404)
def not_found(_):
    return render_template('error.html', code=404, message="We couldn't find that page."), 404


@app.errorhandler(400)
def bad_request(_):
    return render_template('error.html', code=400, message='That request was incomplete. Please try again.'), 400


if __name__ == '__main__':
    app.run(debug=False, port=5000)
