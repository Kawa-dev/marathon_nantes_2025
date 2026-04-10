from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
import os
import gpxpy
import gpxpy.gpx

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'marathon_nantes_2025_features.parquet')
GPX_PATH = os.path.join(BASE_DIR, 'data', 'parcours.gpx')

def vitesse_to_allure(vitesse_kmh):
    if vitesse_kmh <= 0 or pd.isna(vitesse_kmh): return "--:--"
    allure_decimal = 60 / vitesse_kmh
    minutes = int(allure_decimal)
    secondes = int((allure_decimal - minutes) * 60)
    return f"{minutes:02d}:{secondes:02d}"

def safe_rank(val):
    try:
        v = int(val)
        return v if v > 0 else None
    except:
        return None

def extract_category_sex(name):
    try:
        if not isinstance(name, str): return 'SE', 'M'
        suffix = name.split('-')[-1].strip()
        parts = suffix.split()
        if len(parts) >= 2:
            return parts[-2].upper(), parts[-1].upper()
        elif len(parts) == 1:
            s = parts[0]
            return s[:-1].upper(), s[-1].upper()
    except:
        pass
    return 'SE', 'M'

def group_category(cat):
    cat = str(cat).upper().strip()
    if cat in ['CA', 'JU', 'ES', 'SE']: return 'Seniors & Jeunes (<35 ans)'
    if cat in ['M0', 'M1', 'M2', 'V1']: return 'Masters 0-2 (35-49 ans)'
    if cat in ['M3', 'M4', 'V2']: return 'Masters 3-4 (50-59 ans)'
    # Tout le reste (M5, M6...) va ici
    return 'Masters 5+ (60 ans et +)'

def load_gpx_track():
    if not os.path.exists(GPX_PATH): return []
    with open(GPX_PATH, 'r', encoding='utf-8') as gpx_file:
        gpx = gpxpy.parse(gpx_file)
    points = []
    total_dist = 0.0
    previous_point = None
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                if previous_point:
                    total_dist += point.distance_2d(previous_point) / 1000.0 
                ele = point.elevation if point.elevation is not None else 0.0
                points.append({'lat': point.latitude, 'lng': point.longitude, 'dist': total_dist, 'ele': ele})
                previous_point = point
    return points

if os.path.exists(DATA_PATH):
    df = pd.read_parquet(DATA_PATH)
    df = df.fillna(0)
    
    df[['Cat_Brute', 'Sexe']] = df['Nom'].apply(lambda x: pd.Series(extract_category_sex(x)))
    df['Macro_Cat'] = df['Cat_Brute'].apply(group_category)
    
    df_finishers = df[df['Passage_ARRIVEE_sec'] > 0].copy()
    
    total_coureurs = len(df_finishers)
    vitesse_moy = df_finishers['Vitesse_kmh_ARRIVEE'].mean()
    vitesse_med = df_finishers['Vitesse_kmh_ARRIVEE'].median()
    
    allure_moy = vitesse_to_allure(vitesse_moy)
    allure_med = vitesse_to_allure(vitesse_med)
    
    df_finishers['is_negative_split'] = (df_finishers['Passage_ARRIVEE_sec'] - df_finishers['Passage_KM21_sec']) < df_finishers['Passage_KM21_sec']
    nb_neg = len(df_finishers[df_finishers['is_negative_split'] == True])
    murs = len(df_finishers[df_finishers['Derive_Allure_vs_Precedent_%_KM37'] >= 30])
    
    pct_neg = round((nb_neg / total_coureurs) * 100, 1) if total_coureurs > 0 else 0
    pct_murs = round((murs / total_coureurs) * 100, 1) if total_coureurs > 0 else 0
    
    bins_minutes = np.arange(120, 435, 15) 
    labels_tranches = [f"{int(b//60)}h{int(b%60):02d}" for b in bins_minutes[:-1]]
    df_finishers['Minutes_Totales'] = df_finishers['Passage_ARRIVEE_sec'] / 60
    df_finishers['Tranche'] = pd.cut(df_finishers['Minutes_Totales'], bins=bins_minutes, labels=labels_tranches, right=False)
    
    dist_sexe = df_finishers.dropna(subset=['Tranche']).groupby(['Tranche', 'Sexe'], observed=False).size().unstack(fill_value=0)
    hist_labels = dist_sexe.index.tolist()
    hist_hommes = dist_sexe['M'].tolist() if 'M' in dist_sexe else [0]*len(hist_labels)
    hist_femmes = dist_sexe['F'].tolist() if 'F' in dist_sexe else [0]*len(hist_labels)
    
    # --- DISTRIBUTION PAR CATÉGORIE ---
    dist_cat = df_finishers.dropna(subset=['Tranche']).groupby(['Tranche', 'Macro_Cat'], observed=False).size().unstack(fill_value=0)
    # On définit l'ordre exact pour les graphiques
    macro_cats = ['Seniors & Jeunes (<35 ans)', 'Masters 0-2 (35-49 ans)', 'Masters 3-4 (50-59 ans)', 'Masters 5+ (60 ans et +)']
    
    dist_cat_data = {}
    for c in macro_cats:
        dist_cat_data[c] = dist_cat[c].tolist() if c in dist_cat else [0]*len(hist_labels)
    
    # --- RÉPARTITION GLOBALE (DONUT) ---
    sexe_counts = df_finishers['Sexe'].value_counts()
    sexe_labels = sexe_counts.index.tolist()
    sexe_values = sexe_counts.tolist()
    
    # Reindex pour avoir le même ordre que macro_cats
    cat_counts = df_finishers['Macro_Cat'].value_counts().reindex(macro_cats).fillna(0)
    cat_labels = cat_counts.index.tolist()
    cat_values = cat_counts.tolist()
    
    # --- STATS DU MUR AU KM30 (Tranches mathématiques redéfinies) ---
    mur_bins = [-np.inf, -5, 5, 15, 30, np.inf]
    mur_labels_txt = ['< -5%', 'Neutre (-5% à +5%)', '+5% à +15%', '+15% à +30%', '> +30%']
    df_finishers['Mur_Tranche'] = pd.cut(df_finishers['Derive_Allure_vs_Precedent_%_KM37'], bins=mur_bins, labels=mur_labels_txt)
    
    # On s'assure d'avoir toutes les tranches même si elles sont à 0
    mur_counts_serie = df_finishers['Mur_Tranche'].value_counts().reindex(mur_labels_txt).fillna(0)
    mur_labels = mur_counts_serie.index.tolist()
    mur_values = mur_counts_serie.tolist()

    gpx_data = load_gpx_track()
else:
    df = pd.DataFrame()
    gpx_data = []

@app.route('/')
def index():
    if df.empty: return "Fichier de données introuvable."
    return render_template('index.html', 
                           total=total_coureurs, 
                           vitesse_moy=round(vitesse_moy, 2), allure_moy=allure_moy,
                           vitesse_med=round(vitesse_med, 2), allure_med=allure_med,
                           nb_neg=nb_neg, pct_neg=pct_neg,
                           murs=murs, pct_murs=pct_murs,
                           hist_labels=hist_labels, 
                           hist_hommes=hist_hommes, hist_femmes=hist_femmes,
                           dist_cat_data=dist_cat_data, 
                           sexe_labels=sexe_labels, sexe_values=sexe_values,
                           cat_labels=cat_labels, cat_values=cat_values,
                           mur_labels=mur_labels, mur_values=mur_values)

@app.route('/api/gpx')
def get_gpx():
    return jsonify(gpx_data)

@app.route('/api/search')
def search():
    query = request.args.get('q', '').lower()
    if not query or df.empty: return jsonify([])
    mask_nom = df['Nom'].str.lower().str.contains(query, na=False)
    mask_dos = df['Dossard'].astype(str).str.startswith(query, na=False)
    results = df[mask_nom | mask_dos].head(10)[['Dossard', 'Nom', 'ARRIVEE']]
    return jsonify(results.to_dict(orient='records'))

@app.route('/api/replay/<int:dossard>')
def get_replay_data(dossard):
    coureur = df[df['Dossard'] == dossard]
    if coureur.empty: return jsonify({"error": "Non trouvé"}), 404
    c = coureur.iloc[0]
    
    timeline = [
        {"km": 0, "sec": 0},
        {"km": 10, "sec": float(c.get('Passage_KM10_sec', 0))},
        {"km": 15, "sec": float(c.get('Passage_KM15_sec', 0))},
        {"km": 21.1, "sec": float(c.get('Passage_KM21_sec', 0))},
        {"km": 25, "sec": float(c.get('Passage_KM25_sec', 0))},
        {"km": 30, "sec": float(c.get('Passage_KM30_sec', 0))},
        {"km": 37, "sec": float(c.get('Passage_KM37_sec', 0))},
        {"km": 40, "sec": float(c.get('Passage_KM40_sec', 0))},
        {"km": 42.195, "sec": float(c.get('Passage_ARRIVEE_sec', 0))}
    ]
    timeline = [pt for pt in timeline if pt['sec'] >= 0] 

    vitesses = [
        float(c.get('Vitesse_kmh_KM10', 0)), 
        float(c.get('Vitesse_kmh_KM10', 0)),
        float(c.get('Vitesse_kmh_KM15', 0)),
        float(c.get('Vitesse_kmh_KM21', 0)),
        float(c.get('Vitesse_kmh_KM25', 0)),
        float(c.get('Vitesse_kmh_KM30', 0)),
        float(c.get('Vitesse_kmh_KM37', 0)),
        float(c.get('Vitesse_kmh_KM40', 0)),
        float(c.get('Vitesse_kmh_ARRIVEE', 0))
    ]

    evolutions = [0] * len(vitesses)
    for i in range(1, len(vitesses)):
        if vitesses[i-1] > 0:
            evolutions[i] = round(((vitesses[i] - vitesses[i-1]) / vitesses[i-1]) * 100, 1)

    classements = [
        None, 
        safe_rank(c.get('Classement_KM10', 0)),
        safe_rank(c.get('Classement_KM15', 0)),
        safe_rank(c.get('Classement_KM21', 0)),
        safe_rank(c.get('Classement_KM25', 0)),
        safe_rank(c.get('Classement_KM30', 0)),
        safe_rank(c.get('Classement_KM37', 0)),
        safe_rank(c.get('Classement_KM40', 0)),
        safe_rank(c.get('Classement_ARRIVEE', 0))
    ]
    
    nom_propre = str(c['Nom']).split('-')[0].strip()

    return jsonify({
        "Dossard": int(c['Dossard']),
        "Nom": nom_propre,
        "Categorie": str(c.get('Cat_Brute', '')), 
        "Sexe": str(c.get('Sexe', '')),
        "Chrono": str(c['ARRIVEE']),
        "Derive": float(c.get('Derive_Allure_vs_Precedent_%_KM37', 0)),
        "Timeline": timeline,
        "Vitesses": vitesses,
        "Evolutions": evolutions,
        "Classements": classements
    })

if __name__ == '__main__':
    app.run(debug=True)