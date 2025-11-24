# NBA Live Win Probability

Estimate **in-game home win probability** for NBA games.  
Start from **pregame moneyline odds**, then update after each play using live play-by-play.
CatBoost model was trained on **2.5 million samples** that were a mix of numerical, categorical, and text data
odds data are from: https://www.oddsportal.com/basketball/usa/nba/
**Currently working on scaling this and tuning.**
## Quickstart

```bash
# (optional) create a venv
python -m venv venv && source venv/bin/activate

# minimal deps
pip install --upgrade pip
pip install pandas numpy tqdm catboost nba_api

python main.py #requires python 3.10
```
## Sample output
```bash
ORL 0.48 proba vs. NYK 0.52 proba
--------------------------------------------------------------------------------
CHA 0.036 proba vs. LAC 0.96 proba
--------------------------------------------------------------------------------
```