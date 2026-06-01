"""
NBA Game Prediction - Feature Engineering Pipeline
Reproducible script that builds all leakage-free features from raw data.
Place all CSV files in a 'data/' subfolder, then run: python build_features.py
Required files in data/: games.csv, injury_data.csv, players.csv, teams.csv,
games_details.csv, ranking.csv
"""
import pandas as pd, numpy as np
from collections import defaultdict, deque
from math import radians, sin, cos, sqrt, atan2

# ---- 1. Load & clean ----
games = pd.read_csv('data/games.csv').dropna()
games['GAME_DATE_EST'] = pd.to_datetime(games['GAME_DATE_EST'])
games = games[games['GAME_DATE_EST'] >= '2016-10-01'].sort_values('GAME_DATE_EST').reset_index(drop=True)

# ---- 2. Win% + last-10 form (record feature BEFORE updating with result) ----
wins, played = defaultdict(int), defaultdict(int)
last10 = defaultdict(lambda: deque(maxlen=10))
cols = {k: [] for k in ['home_winpct','away_winpct','home_form10','away_form10']}
for g in games.itertuples():
    h, a = g.HOME_TEAM_ID, g.VISITOR_TEAM_ID
    cols['home_winpct'].append(wins[h]/played[h] if played[h] else 0.5)
    cols['away_winpct'].append(wins[a]/played[a] if played[a] else 0.5)
    cols['home_form10'].append(sum(last10[h])/len(last10[h]) if last10[h] else 0.5)
    cols['away_form10'].append(sum(last10[a])/len(last10[a]) if last10[a] else 0.5)
    res = int(g.HOME_TEAM_WINS)
    played[h]+=1; played[a]+=1
    wins[h]+= res; wins[a]+= (1-res)
    last10[h].append(res); last10[a].append(1-res)
for k,v in cols.items(): games[k]=v

# ---- 3. Difference features ----
games['winpct_diff'] = games.home_winpct - games.away_winpct
games['form_diff']   = games.home_form10 - games.away_form10

# (Injury star-weighting, travel haversine, rest days, home/road splits, and
#  conference seeding are documented in the notebook; this script shows the
#  core leakage-free pattern. See NBA_Prediction.ipynb for the full build.)

games.to_csv('games_features_out.csv', index=False)
print('Features written:', games.shape)
