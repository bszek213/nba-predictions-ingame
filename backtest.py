import pandas as pd
import glob
import matplotlib.pyplot as plt
from scipy.stats import zscore
import numpy as np
from tqdm import tqdm

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

def plot_backtest(game_df, home_preds, away_preds, cm_metric, cm_label="CM metric"):
    x = game_df['SECONDS_FROM_START'].to_numpy()
    # print("monotonic?", np.all(np.diff(x) >= 0))
    fig, ax1 = plt.subplots(figsize=(10, 5))
    # --- Left y-axis: probabilities ---
    l1, = ax1.plot(x, home_preds, label=f"Home {game_df['HOME_TEAM_ABBREV'].iloc[0]}")
    l2, = ax1.plot(x, away_preds, label=f"Away {game_df['AWAY_TEAM_ABBREV'].iloc[0]}")
    ax1.set_xlabel("Seconds from Start")
    ax1.set_ylabel("Predicted Win Probability")
    ax1.set_ylim(0, 1)
    ax2 = ax1.twinx()
    l3, = ax2.plot(x, cm_metric, color='black', linestyle="-", label=cm_label)
    ax2.set_ylabel(cm_label)
    lines = [l1, l2, l3]
    labels = [ln.get_label() for ln in lines]
    ax1.legend(lines, labels, loc="best")
    plt.tight_layout()
    plt.show()

# def chaos_metric(preds):
#     """
#     C(t)=z(vol(t))+z(flip(t))+z(jerk(t))
#     """
#     volatility = np.nanmean(zscore(pd.Series(preds).rolling(window=10,min_periods=2).std(), nan_policy='omit'))
#     flips_norm = sum((preds[i] > 0.5) != (preds[i-1] > 0.5) for i in range(1, len(preds))) / len(preds)
#     jerk = np.nanmean(zscore(pd.Series(preds).diff().abs(), nan_policy='omit'))
#     return volatility + flips_norm + jerk

def rolling_components(preds, window=10):
    preds = np.asarray(preds, dtype=float)
    p_lead = np.maximum(preds, 1 - preds)

    #Rolling volatility of leader prob
    vol_t = pd.Series(p_lead).rolling(window, min_periods=2).std()

    #Flip-rate (leader changes) in rolling window
    flips_step = (preds[1:] > 0.5) != (preds[:-1] > 0.5)  # per-step flips
    flip_rate_t = pd.Series(np.r_[0, flips_step.astype(int)]).rolling(window, min_periods=2).mean()

    #Rolling jerkiness (mean abs step change)
    jerk_step = np.abs(np.diff(p_lead, prepend=p_lead[0]))
    jerk_t = pd.Series(jerk_step).rolling(window, min_periods=2).mean()

    return {
        "p_lead": p_lead,
        "vol_t": vol_t,
        "flip_rate_t": flip_rate_t,
        "jerk_t": jerk_t
    }

def fit_cm_norm_params(games_df, model, X_cols, window=10):
    all_vol, all_flip, all_jerk = [], [], []

    for gid in tqdm(games_df["GAME_ID"].unique()):
        g = games_df[games_df["GAME_ID"] == gid]
        Xg = g[X_cols]
        preds = model.predict_proba(Xg)[:, 1]

        comps = rolling_components(preds, window=window)
        all_vol.extend(comps["vol_t"].dropna().values)
        all_flip.extend(comps["flip_rate_t"].dropna().values)
        all_jerk.extend(comps["jerk_t"].dropna().values)

    mu_sigma = {
        "vol": (np.mean(all_vol), np.std(all_vol) + 1e-9),
        "flip_rate": (np.mean(all_flip), np.std(all_flip) + 1e-9),
        "jerk": (np.mean(all_jerk), np.std(all_jerk) + 1e-9),
    }
    return mu_sigma

def cm_game(preds, mu_sigma, window=10):
    comps = rolling_components(preds, window=window)

    vol_t = comps["vol_t"]
    flip_t = comps["flip_rate_t"]
    jerk_t = comps["jerk_t"]

    z_vol  = (vol_t  - mu_sigma["vol"][0]) / mu_sigma["vol"][1]
    z_flip = (flip_t - mu_sigma["flip_rate"][0]) / mu_sigma["flip_rate"][1]
    z_jerk = (jerk_t - mu_sigma["jerk"][0]) / mu_sigma["jerk"][1]

    cm_t = (z_vol + z_flip + z_jerk).fillna(0).to_numpy()
    return cm_t

def build_cm_distribution(games_df, model, X_cols, mu_sigma, window=10):
    all_cm = []
    for gid in tqdm(games_df["GAME_ID"].unique()):
        g = games_df[games_df["GAME_ID"] == gid].sort_values("SECONDS_FROM_START")
        preds = model.predict_proba(g[X_cols])[:, 1]
        cm_t = cm_game(preds, mu_sigma, window=window)
        all_cm.extend(cm_t)
    plt.figure()
    plt.hist(all_cm,bins=400,color='skyblue')
    plt.savefig('cm_hist.png',dpi=400)
    plt.close()
    return np.asarray(all_cm)

def first_bet_time(game_df, home_preds, cm_t,
                   cm_cutoff, p_thresh=0.50,
                   consec_K=5, time_min=None, time_max=None):
    away_preds = 1 - home_preds
    p_lead = np.maximum(home_preds, away_preds)
    leader_is_home = home_preds >= away_preds

    stable = cm_t <= cm_cutoff

    # require K consecutive stable points
    run = 0
    for t in range(len(cm_t)):
        sec = game_df["SECONDS_FROM_START"].iloc[t]
        if time_min is not None and sec < time_min:
            run = 0
            continue
        if time_max is not None and sec > time_max:
            break

        if stable[t]:
            run += 1
        else:
            run = 0

        if run >= consec_K and p_lead[t] >= p_thresh:
            bet_team = "home" if leader_is_home[t] else "away"
            return t, bet_team

    return None, None

def first_bet_time(game_df, home_preds, cm_t,
                   cm_cutoff, p_thresh=0.70,
                   consec_K=5, time_min=None, time_max=None):
    """
    Returns (bet_idx, bet_team) or (None, None)
    """
    away_preds = 1 - home_preds
    p_lead = np.maximum(home_preds, away_preds)
    leader_is_home = home_preds >= away_preds

    stable = cm_t <= cm_cutoff

    # require K consecutive stable points
    run = 0
    for t in range(len(cm_t)):
        sec = game_df["SECONDS_FROM_START"].iloc[t]
        if time_min is not None and sec < time_min:
            run = 0
            continue
        if time_max is not None and sec > time_max:
            break

        if stable[t]:
            run += 1
        else:
            run = 0

        if run >= consec_K and p_lead[t] >= p_thresh:
            bet_team = "home" if leader_is_home[t] else "away"
            return t, bet_team

    return None, None

def backtest_once_per_game(games_df, model, X_cols, mu_sigma,
                           cm_cutoff, p_thresh=0.50, consec_K=5,
                           time_min=None, time_max=None,
                           target_col="TARGET", stake=100.0, window=10):
    results = []

    for gid in games_df["GAME_ID"].unique():
        g = games_df[games_df["GAME_ID"] == gid].sort_values("SECONDS_FROM_START")
        home_preds = model.predict_proba(g[X_cols])[:, 1]
        cm_t = cm_game(home_preds, mu_sigma, window=window)

        bet_idx, bet_team = first_bet_time(
            g, home_preds, cm_t,
            cm_cutoff=cm_cutoff,
            p_thresh=p_thresh,
            consec_K=consec_K,
            time_min=time_min,
            time_max=time_max
        )

        if bet_idx is None:
            results.append({"GAME_ID": gid, "bet": False})
            continue

        home_won = int(g[target_col].iloc[0]) == 1
        bet_won = (bet_team == "home" and home_won) or (bet_team == "away" and not home_won)
        profit = stake if bet_won else -stake

        results.append({
            "GAME_ID": gid,
            "bet": True,
            "bet_team": bet_team,
            "bet_time": g["SECONDS_FROM_START"].iloc[bet_idx],
            "p_at_bet": home_preds[bet_idx] if bet_team=="home" else 1-home_preds[bet_idx],
            "cm_at_bet": cm_t[bet_idx],
            "bet_won": bet_won,
            "profit": profit
        })

    res_df = pd.DataFrame(results)
    bets = res_df[res_df.bet]
    summary = {
        "n_games": len(res_df),
        "n_bets": len(bets),
        "bet_rate": len(bets)/len(res_df),
        "hit_rate": bets.bet_won.mean() if len(bets) else 0,
        "avg_profit": bets.profit.mean() if len(bets) else 0,
        "roi_per_bet": bets.profit.mean()/stake if len(bets) else 0
    }
    return res_df, summary

def backtest_games(model):
    all_files = glob.glob("betting_data/*_betting_data.csv")
    combine_df = pd.DataFrame()
    for file in all_files:
        df = pd.read_csv(file)
        combine_df = pd.concat([combine_df, df], ignore_index=True)

    combine_df, X_cols, CAT_COLS, TEXT_COLS, TARGET, GROUPS = create_features(combine_df)

    mu_sigma = fit_cm_norm_params(combine_df, model, X_cols)
    cm_dist = build_cm_distribution(combine_df,model,X_cols,mu_sigma)

    pd.Series(mu_sigma).to_json("mu_sigma.json")
    pd.Series(cm_dist).to_csv('betting_data/cm_dist.csv',index=False)
    best = None
    for perc in tqdm([10,20,30,40,50],desc='Percentile'):
        cm_cut = np.percentile(cm_dist, perc)
        for p_th in tqdm(np.linspace(0.6,0.85,6),desc='P threshold'):
            for K in tqdm([3,5,7], desc='K threshold'):
                _, summ = backtest_once_per_game(
                    combine_df, model, X_cols, mu_sigma,
                    cm_cutoff=cm_cut, p_thresh=p_th, consec_K=K,
                    time_min=600, time_max=2600,
                    target_col=TARGET
                )
                if best is None or summ["avg_profit"] > best["avg_profit"]:
                    best = {"perc": perc, "p_thresh": p_th, "K": K, **summ}
    print(best)
    # for id in combine_df['GAME_ID'].unique():
    #     game_df = combine_df[combine_df['GAME_ID'] == id]
    #     game_df = game_df.sort_values(by=['SECONDS_FROM_START'])
    #     X_game = game_df[X_cols]
    #     home_preds = model.predict_proba(X_game)[:, 1]
    #     cm_t = cm_game(home_preds, mu_sigma, window=10)
    #     away_preds = 1 - home_preds
    #     plot_backtest(game_df, home_preds, away_preds, cm_t)