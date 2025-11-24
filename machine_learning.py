import pandas as pd
from sklearn.model_selection import train_test_split
import glob
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score, accuracy_score
from sklearn.calibration import CalibratedClassifierCV
from catboost import CatBoostClassifier, Pool
import os
from pbp_collect import _build_game_meta_this_season, _expand_pbp_for_game, build_today_live_meta
from nba_api.stats.endpoints import LeagueGameFinder, PlayByPlayV3
from nba_api.live.nba.endpoints import playbyplay as live_pbp
import json, re
from backtest import cm_game
from time import sleep
def ml_to_prob(series: pd.Series):
    s = series.values
    # raw implied probability from American moneyline
    return np.where(s < 0, (-s) / ((-s) + 100.0), 100.0 / (s + 100.0))

def convert_date_to_cycle(df):
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

    m = df["GAME_DATE"].dt.month
    d = df["GAME_DATE"].dt.day  
    df["month_sin"] = np.sin(2*np.pi*(m-1) / 12)
    df["month_cos"] = np.cos(2*np.pi*(m-1) / 12)
    df["dom_sin"] = np.sin(2*np.pi*(d-1) / 31)
    df["dom_cos"] = np.cos(2*np.pi*(d-1) / 31)

    # (Nice extra) day-of-week cyclical:
    dow = df["GAME_DATE"].dt.weekday  # 0=Mon
    df["dow_sin"] = np.sin(2*np.pi*dow / 7)
    df["dow_cos"] = np.cos(2*np.pi*dow / 7)
    return df

def create_features(combine_df):
    cols = ["Team Odd", "Opposition Odd"]
    combine_df[cols] = combine_df[cols].apply(pd.to_numeric, errors="coerce")
    p_team_raw = ml_to_prob(combine_df["Team Odd"])
    p_opp_raw  = ml_to_prob(combine_df["Opposition Odd"])
    den = p_team_raw + p_opp_raw
    combine_df["pregame_p_team"] = p_team_raw / den
    combine_df["pregame_p_opp"]  = p_opp_raw  / den

    combine_df = convert_date_to_cycle(combine_df)

    combine_df["t_frac"] = (combine_df["SECONDS_REMAINING_REG"].clip(lower=0) / (48*60)).astype(float)
    combine_df["lead_x_time"]  = combine_df["LEAD_HOME"] * combine_df["t_frac"]
    combine_df["prior_x_time"] = combine_df["pregame_p_team"] * combine_df["t_frac"]  # or pregame_p_home
    # --------
    # extract features
    # --------
    NUM_COLS = [
        "SECONDS_FROM_START", "SECONDS_REMAINING_REG",
        "HOME_PTS", "AWAY_PTS", "LEAD_HOME",
        "actionNumber", "period",
        "pregame_p_opp", "pregame_p_team",'month_sin','month_cos',
        'dom_sin','dom_cos','dow_sin','dow_cos','t_frac','lead_x_time','prior_x_time'
    ]

    CAT_COLS = [
        "HOME_TEAM_ABBREV", "AWAY_TEAM_ABBREV",
        # "teamTricode",
        "actionType", "subType"
    ]

    TEXT_COLS = ["description"]  #  current event text (post-play is OK)

    TARGET = "HOME_WIN"
    GROUPS = "GAME_ID"
    
    #keep only columns that actually exist
    present = set(combine_df.columns)
    NUM_COLS  = [c for c in NUM_COLS  if c in present]
    CAT_COLS  = [c for c in CAT_COLS  if c in present]
    TEXT_COLS = [c for c in TEXT_COLS if c in present]

    X_cols = list(dict.fromkeys(NUM_COLS + CAT_COLS + TEXT_COLS))  # dedupe, preserve order

    #sanitize dtypes / NaNs
    for c in NUM_COLS:
        combine_df[c] = pd.to_numeric(combine_df[c], errors="coerce").fillna(0.0)
    for c in CAT_COLS:
        combine_df[c] = combine_df[c].astype(str).fillna("UNK")
    for c in TEXT_COLS:
        combine_df[c] = combine_df[c].astype(str).fillna("")

    # optional: drop rows without odds
    if "pregame_p_team" in present and "pregame_p_opp" in present:
        combine_df = combine_df.dropna(subset=["pregame_p_team","pregame_p_opp"])
    
    return combine_df, X_cols, CAT_COLS, TEXT_COLS, TARGET, GROUPS

def train_models():
    #import all csvs with betting_data in it
    all_files = glob.glob("betting_data//*_betting_data.csv")
    combine_df = pd.DataFrame()
    for file in all_files:
        df = pd.read_csv(file)
        combine_df = pd.concat([combine_df, df], ignore_index=True)
        # print(df.head())
        # print(80 * '-')
    
    combine_df, X_cols, CAT_COLS, TEXT_COLS, TARGET, GROUPS = create_features(combine_df)

    # 6) grouped split by game
    groups = combine_df[GROUPS].astype(str).values
    y = combine_df[TARGET].astype(int).values
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, va_idx = next(gss.split(combine_df[X_cols], y, groups))

    df_tr, df_va = combine_df.iloc[tr_idx].copy(), combine_df.iloc[va_idx].copy()
    y_tr, y_va = y[tr_idx], y[va_idx]

    # 7) per-game row weights (prevents long games from dominating)
    rows_per_game_tr = df_tr.groupby(GROUPS).size()
    df_tr["w"] = 1.0 / df_tr[GROUPS].map(rows_per_game_tr)
    rows_per_game_va = df_va.groupby(GROUPS).size()
    df_va["w"] = 1.0 / df_va[GROUPS].map(rows_per_game_va)

    print(f"Train shape: {df_tr.shape}, Valid shape: {df_va.shape}")
    # 8) build CatBoost pools
    cat_idx  = [X_cols.index(c) for c in CAT_COLS if c in X_cols]
    text_idx = [X_cols.index(c) for c in TEXT_COLS if c in X_cols]

    train_pool = Pool(df_tr[X_cols], y_tr, weight=df_tr["w"],
                      cat_features=cat_idx, text_features=text_idx)
    valid_pool = Pool(df_va[X_cols], y_va, weight=df_va["w"],
                      cat_features=cat_idx, text_features=text_idx)

    if not os.path.exists("models/winprob_catboost.cbm"):
        # 9) model (regularized + early stopping; select by Logloss)
        model = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="Logloss",
            custom_metric=["AUC","BrierScore"],
            learning_rate=0.01,
            depth=7,
            l2_leaf_reg=100,
            subsample=0.7,
            colsample_bylevel=0.6,
            min_data_in_leaf=5000,
            iterations=10000,
            od_type="Iter",
            od_wait=200,
            use_best_model=True,
            # task_type="CPU",
            # thread_count=-1,
            random_seed=123,
            verbose=100,
        )

        print(f"dataset shape: {combine_df.shape}")
        model.fit(train_pool, eval_set=valid_pool)

        # 10) eval
        p_val = model.predict_proba(valid_pool)[:, 1]
        print("Val log loss:", log_loss(y_va, p_val))
        print("Val brier:", brier_score_loss(y_va, p_val))
        print("Val AUC:", roc_auc_score(y_va, p_val))
        print(f'Accuracy: {accuracy_score(y_va, (p_val>=0.2).astype(int))}')
        thresholds = np.linspace(0, 1, 50)
        accuracies = []
        for t in thresholds:
            preds = (p_val >= t).astype(int)
            acc = accuracy_score(y_va, preds)
            accuracies.append(acc)
        t_star = thresholds[int(np.argmax(accuracies))]
        print(f'Best threshold: {t_star}')
        # Plot Accuracy vs. Threshold
        plt.figure(figsize=(7, 4))
        plt.plot(thresholds, accuracies, marker='o')
        plt.title('Accuracy vs. Threshold')
        plt.xlabel('Threshold')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('accuracy_vs_threshold.png')

        os.makedirs("models", exist_ok=True)
        model.save_model("models/winprob_catboost.cbm")
        pd.Series(X_cols).to_json("models/winprob_columns.json", orient="values")
    else:
        #load model
        model = CatBoostClassifier()
        model.load_model("models/winprob_catboost.cbm")
        X_cols = pd.read_json("models/winprob_columns.json", orient="values").squeeze().tolist()
    
    return model

def get_latest_action_row(game_id: str, home_abbr=None, away_abbr=None):
    live = live_pbp.PlayByPlay(game_id=game_id).get_dict()
    actions = live.get("game", {}).get("actions", [])
    if not actions:
        return None  # game not started or feed empty

    a = actions[-1]  # last (most recent) action

    # robust int cast for scores (strings/blank -> 0)
    def to_int(x):
        try:
            return int(float(x))
        except (TypeError, ValueError):
            return 0

    home_pts = to_int(a.get("scoreHome"))
    away_pts = to_int(a.get("scoreAway"))

    return {
        "GAME_ID": live.get("game", {}).get("gameId"),
        # "HOME_TEAM_ABBREV": home_abbr,
        # "AWAY_TEAM_ABBREV": away_abbr,
        "actionNumber": int(a.get("actionNumber")),
        "period": int(a.get("period")),
        "clock": a.get("clock"),
        # "teamTricode": a.get("teamTricode"),
        "actionType": a.get("actionType"),
        "subType": a.get("subType"),
        "description": a.get("description"),
        "HOME_PTS": home_pts,
        "AWAY_PTS": away_pts,
        "LEAD_HOME": home_pts - away_pts,
    }

def iso_clock_to_secs(clock: str, period: int) -> int:
    # 'PT04M53.00S' -> 293
    m = re.match(r"^PT(?:(\d+)M)?(?:(\d+(?:\.\d+)?)S)?$", clock or "")
    mins = int(m.group(1) or 0) if m else 0
    secs = float(m.group(2) or 0) if m else 0.0
    s_rem = int(round(mins*60 + secs))
    if period <= 4:
        seconds_from_start = (period-1)*720 + (720 - s_rem)
    else:
        seconds_from_start =  2880 + (period-5)*300 + (300 - s_rem)
    seconds_remaining_reg = max(0, 2880 - seconds_from_start)
    return seconds_from_start, seconds_remaining_reg

def ml_to_prob_individual(ml):
    ml = float(ml)
    return (-ml) / ((-ml) + 100.0) if ml < 0 else 100.0 / (ml + 100.0)

def input_pregame_probabilities(home_team_abbr, away_team_abbr):
    df_probs = pd.read_csv("curr_pregame_proba.csv")
    home_row = df_probs[(df_probs['home_abbrev'] == home_team_abbr)]
    away_row = df_probs[(df_probs['away_abbrev'] == away_team_abbr)]
    home = float(home_row['home_moneyline'].values[0])
    away = float(away_row['away_moneyline'].values[0])
    # home_str = input(f"Enter {home_team_abbr} moneyline (e.g., -145): ").strip()
    # away_str = input(f"Enter {away_team_abbr} moneyline (e.g., +125): ").strip()
    # home = float(home_str)   #handles "+125" and whitespace
    # away = float(away_str)
    p_home_raw = ml_to_prob_individual(home)
    p_away_raw = ml_to_prob_individual(away)
    den = p_home_raw + p_away_raw
    pregame_p_team = p_home_raw / den
    pregame_p_opp = p_away_raw / den

    return pregame_p_team, pregame_p_opp
def predict_win_probability(model):
    """
    threshold with highest accuracy: 0.4897959183673469
    bestTest = 0.4699351563
    bestIteration = 177

    Shrink model to first 178 iterations.
    Val log loss: 0.47335510922643803
    Val brier: 0.15607425591681295
    Val AUC: 0.8532404389391786
    Accuracy: 0.6707194267285584
    Best threshold: 0.4897959183673469

    {'perc': 10, 'p_thresh': 0.85, 'K': 7, 'n_games': 7053, 'n_bets': 2161, 
    'bet_rate': 0.30639444208138383, 'hit_rate': 0.974548819990745,
    'avg_profit': 0.9490976399814901, 'roi_per_bet': 0.9490976399814901}
    """
    # ---- hyperparameters (from backtest) ----
    perc = 16#10 one std...
    p_thresh = 0.85
    K = 7
    min_points = 10
    perc = 10
    # df = _build_game_meta_this_season()
    mu_sigma = pd.read_json("mu_sigma.json").to_dict()
    cm_dist_arr = pd.read_csv("cm_dist.csv").values.ravel()  # 1D array
    with open('models/winprob_columns.json', 'r') as file:
            X_cols = json.load(file)
    cm_cut = np.percentile(cm_dist_arr, perc)
    game_state = {}

    NUM_COLS = [
        "SECONDS_FROM_START", "SECONDS_REMAINING_REG",
        "HOME_PTS", "AWAY_PTS", "LEAD_HOME",
        "actionNumber", "period",
        "pregame_p_opp", "pregame_p_team",'month_sin','month_cos',
        'dom_sin','dom_cos','dow_sin','dow_cos','t_frac','lead_x_time','prior_x_time'
    ]

    CAT_COLS = [
        "HOME_TEAM_ABBREV", "AWAY_TEAM_ABBREV",
        "teamTricode","actionType", "subType"
    ]

    TEXT_COLS = ["description"]  #  current event text (post-play is OK)
    
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RESET = "\033[0m"

    while True:
        df = build_today_live_meta()

        #get games today
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'], utc=True).dt.tz_convert('US/Eastern').dt.tz_localize(None)
        df = df[df['STATUS'] >= 2]  # games that have started
        df = df[~df['STATUS_TEXT'].str.contains('Final', na=False)]  # exclude final games

        mu_sigma = pd.read_json("mu_sigma.json").to_dict()
        cm_dist = pd.read_csv("cm_dist.csv")

        cat_idx  = [X_cols.index(c) for c in CAT_COLS if c in X_cols]
        text_idx = [X_cols.index(c) for c in TEXT_COLS if c in X_cols]
        df = convert_date_to_cycle(df)
        live_ids = set(df["GAME_ID"].astype(str).unique())
        for i, row in df.iterrows():
            game_id = str(row["GAME_ID"])
            latest_play = get_latest_action_row(game_id)
            # with open("live_pbp.json", "w") as f:
            #     json.dump(live, f, indent=2, ensure_ascii=False)
            iss, s_rem = iso_clock_to_secs(latest_play["clock"], latest_play["period"])
            latest_play["SECONDS_FROM_START"] = iss
            latest_play["SECONDS_REMAINING_REG" ] = s_rem
            latest_play["pregame_p_team"], latest_play["pregame_p_opp"] = input_pregame_probabilities(row['HOME_TEAM_ABBREV'], 
                                                                                                    row['AWAY_TEAM_ABBREV'])
            seconds_remaining = max(latest_play["SECONDS_REMAINING_REG"], 0)  #clip at 0
            latest_play["t_frac"] = float(seconds_remaining) / (48 * 60)
            latest_play["lead_x_time"]  = float(latest_play["LEAD_HOME"] * latest_play["t_frac"])
            latest_play["prior_x_time"] = float(latest_play["pregame_p_team"] * latest_play["t_frac"])  # or pregame_p_home
            
            latest_df = pd.DataFrame([latest_play])
            row_df = row.to_frame().T.reset_index(drop=True)
            combine_df = pd.concat([row_df, latest_df], axis=1)
            combine_df = combine_df.loc[:, ~combine_df.columns.duplicated()].copy() #check this in the future if you change anything

            pool = Pool(combine_df[X_cols], cat_features=cat_idx, text_features=text_idx)
            p = model.predict_proba(pool)[:, 1]
            home_proba = p[0]
            away_proba = 1.0 - home_proba
            
            #UPDATE BASED ON ACTION
            st = game_state.setdefault(game_id, {
                "preds": [],
                "below_run": 0,
                "bet_made": False,
                "last_action": None
            })
            action_num = latest_play.get("actionNumber", None)
            if action_num is not None and st["last_action"] == action_num:
                # no new play dont append
                continue
            st["last_action"] = action_num

            st["preds"].append(home_proba)
            print(f"{GREEN}{combine_df['HOME_TEAM_ABBREV'].iloc[0]} {home_proba:.2g} proba vs. {combine_df['AWAY_TEAM_ABBREV'].iloc[0]} {away_proba:.2g} proba{RESET}")

            #betting rule
            if (not st["bet_made"]) and (len(st["preds"]) >= min_points):
                cm_t = cm_game(st["preds"], mu_sigma, window=10)
                cm_now = cm_t[-1]
                # stability run
                if cm_now <= cm_cut:
                    st["below_run"] += 1
                else:
                    st["below_run"] = 0

                print(f"chaos metric now: {cm_now:.2g} || num times below percentile (<{cm_cut:.2g}): {st['below_run']}")
                print(40 * '-')

                p_lead_now = max(home_proba, away_proba)
                leader_is_home = home_proba >= away_proba

                if st["below_run"] >= K: #and p_lead_now >= p_thresh
                    st["bet_made"] = True
                    bet_team = (combine_df['HOME_TEAM_ABBREV'].iloc[0]
                                if leader_is_home else combine_df['AWAY_TEAM_ABBREV'].iloc[0])

                    print(f"{YELLOW} BET TRIGGERED on {bet_team} "
                          f"(p_lead={p_lead_now:.2g}, cm={cm_now:.2f}, run={st['below_run']}){RESET}")
        for gid in list(game_state.keys()):
            if gid not in live_ids:
                del game_state[gid]
        print(f"{YELLOW}----------------------------{RESET}")
        sleep(10)