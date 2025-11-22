import time
import pandas as pd
from tqdm import tqdm
# from nba_api.stats.endpoints import (
#     LeagueGameFinder,
#     BoxScoreAdvancedV2,
#     BoxScoreTraditionalV2,
#     BoxScoreFourFactorsV2,
#     BoxScoreMiscV2,
#     BoxScoreScoringV2,
# )
from nba_api.stats.endpoints import (
    LeagueGameFinder,
    BoxScoreAdvancedV3,
    BoxScoreTraditionalV3,
    BoxScoreFourFactorsV3,
    BoxScoreMiscV3,
    BoxScoreScoringV3,
)
from datetime import datetime
import numpy as np
import os
from urllib3.exceptions import ReadTimeoutError
import requests
import json

ALL_TEAMS = ['PHI', 'BOS', 'CHA', 'POR', 'TOR', 'DAL', 'GSW', 'MIN', 'UTA',
 'MIA', 'SAS', 'NOP', 'CHI', 'DEN', 'LAL', 'LAC', 'ORL', 'NYK', 'WAS', 'ATL',
 'PHX', 'SAC', 'CLE', 'HOU', 'MEM', 'BKN', 'MIL', 'OKC', 'DET', 'IND'
 ]

def _parse_date_to_dt(s):
    return datetime.strptime(s, "%Y-%m-%d")

# def _add_calendar_features(df):
#     df = df.sort_values("GAME_DATE").reset_index(drop=True)
#     df["GAME_DATE_DT"] = df["GAME_DATE"].apply(_parse_date_to_dt)
#     df["days_rest"] = df["GAME_DATE_DT"].diff().dt.days
#     df["days_rest"] = df["days_rest"].fillna(7)  # first game
#     df["is_b2b"] = (df["days_rest"] == 1).astype(int)
#     df["is_home"] = (~df["MATCHUP"].str.contains("@")).astype(int)
#     return df
def _add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    # normalize to datetime, drop tz if any
    df["GAME_DATE_DT"] = pd.to_datetime(df["GAME_DATE"], errors="coerce").dt.tz_localize(None)
    df["GAME_YEAR"] = df["GAME_DATE_DT"].dt.year
    df["GAME_MONTH"] = df["GAME_DATE_DT"].dt.month
    df["GAME_DAY"] = df["GAME_DATE_DT"].dt.day
    df["GAME_DOW"] = df["GAME_DATE_DT"].dt.dayofweek  # 0=Mon
    return df

def collect_team_data(team_abbreviation_1, team_abbreviation_2, season="2024-25"):
    # load your target feature list
    # all_cols = pd.read_csv("data/feature_names.csv")["0"].tolist()
    # print(f"all features: {all_cols}")

    # 1) get ALL games in that season
    gamefinder = LeagueGameFinder(
        season_nullable=season,
        team_id_nullable=None,
        league_id_nullable="00"
    )
    games = gamefinder.get_data_frames()[0]

    team_games_1 = games[games["TEAM_ABBREVIATION"] == team_abbreviation_1].iloc[::-1].reset_index(drop=True)
    team_games_2 = games[games["TEAM_ABBREVIATION"] == team_abbreviation_2].iloc[::-1].reset_index(drop=True)

    print(f"Found {len(team_games_1)} games for {team_abbreviation_1}")
    print(f"Found {len(team_games_2)} games for {team_abbreviation_2}")

    # 2) build DF for team 1
    team1_rows = []
    for game_id, row in tqdm(zip(team_games_1["GAME_ID"], team_games_1.itertuples()), total=len(team_games_1)):
        game_df = _get_game_all_tables(game_id, team_abbreviation_1)
        # attach game-level fields (WL, GAME_DATE, MATCHUP, etc.)
        meta = pd.DataFrame([row._asdict()])
        # _asdict gives Index, so drop it
        meta = meta.drop(columns=["Index"], errors="ignore")
        full_row = pd.concat([meta.reset_index(drop=True), game_df], axis=1)
        team1_rows.append(full_row)
        time.sleep(0.6)  # be nice to API

    team_1_df = pd.concat(team1_rows, ignore_index=True)
    team_1_df = _add_calendar_features(team_1_df)

    # 3) build DF for team 2
    team2_rows = []
    for game_id, row in tqdm(zip(team_games_2["GAME_ID"], team_games_2.itertuples()), total=len(team_games_2)):
        game_df = _get_game_all_tables(game_id, team_abbreviation_2)
        meta = pd.DataFrame([row._asdict()]).drop(columns=["Index"], errors="ignore")
        full_row = pd.concat([meta.reset_index(drop=True), game_df], axis=1)
        team2_rows.append(full_row)
        time.sleep(0.6)

    team_2_df = pd.concat(team2_rows, ignore_index=True)
    team_2_df = _add_calendar_features(team_2_df)

    # 4) line up lengths (your logic)
    if len(team_1_df) != len(team_2_df):
        m = min(len(team_1_df), len(team_2_df))
        team_1_df = team_1_df.iloc[-m:].reset_index(drop=True)
        team_2_df = team_2_df.iloc[-m:].reset_index(drop=True)

    # 5) suffix opponent (team 2) cols
    team_2_df = team_2_df.add_prefix("opp_")

    # 6) concat side by side
    final_df = pd.concat([team_1_df.reset_index(drop=True), team_2_df.reset_index(drop=True)], axis=1)

    # 7) your feature engineering
    final_df = engineer_features(final_df)

    #add time series
    final_df = time_series_engineer(final_df)

    # # 8) select the exact model feature order
    # final_df = final_df[all_cols]

    # drop WL if you don’t want leakage
    return final_df#.drop(columns=["WL"], errors="ignore")

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ---------- 1. minutes → OT flag ----------
    # prefer the v3-style "team_minutes" if present, else fall back to TEAM_MIN
    if "team_minutes" in df.columns:
        mins = df["team_minutes"].astype(str).str.split(":").str[0].astype(int)
    elif "TEAM_MIN" in df.columns:  # sometimes "240" not "240:00"
        mins = df["TEAM_MIN"].astype(str).str.split(":").str[0].astype(int)
    else:
        mins = pd.Series(240, index=df.index)  # default
    df["was_ot"] = (mins > 240).astype(int)

    # to avoid ZeroDivisionError
    poss_team = df["team_possessions"].replace(0, 1)
    poss_opp  = df["opp_possessions"].replace(0, 1)

    # ---------- 2. Offensive / Defensive efficiency ----------
    # OFF = points per own poss
    df["OFF_EFF"] = df["team_points"] / poss_team
    # DEF = opp points per our poss (you could use opp_possessions instead)
    df["DEF_EFF"] = df["opp_points"] / poss_team

    # ---------- 3. Four-factor style differentials ----------
    # eFG%
    df["EFG_DIFF"] = (
        df["team_effectiveFieldGoalPercentage"]
        - df["opp_effectiveFieldGoalPercentage"]
    )

    # TOV%  (you already have these from four-factors)
    df["TOV_DIFF"] = (
        df["team_teamTurnoverPercentage"]
        - df["opp_teamTurnoverPercentage"]
    )

    # OREB%
    df["OREB_DIFF"] = (
        df["team_offensiveReboundPercentage"]
        - df["opp_offensiveReboundPercentage"]
    )

    # FT rate
    df["FT_RATE"] = df["team_freeThrowAttemptRate"]
    df["FT_RATE_opp"] = df["opp_freeThrowAttemptRate"]
    df["FT_RATE_DIFF"] = df["FT_RATE"] - df["FT_RATE_opp"]

    # ---------- 4. Shooting (recomputed from basic) ----------
    # use v3-style names
    df["team_eFG_from_box"] = (
        df["team_fieldGoalsMade"] + 0.5 * df["team_threePointersMade"]
    ) / df["team_fieldGoalsAttempted"].replace(0, 1)

    df["team_TS_PCT"] = (
        df["team_points"]
        / (
            2
            * (
                df["team_fieldGoalsAttempted"]
                + 0.44 * df["team_freeThrowsAttempted"]
            ).replace(0, 1)
        )
    )

    # ---------- 5. Possession-based ----------
    df["PTS_PER_POSS"] = df["team_points"] / poss_team
    df["AST_PER_POSS"] = df["team_assists"] / poss_team
    df["TOV_PER_POSS"] = df["team_turnovers"] / poss_team

    # ---------- 6. Defensive-ish ----------
    # opp 2PA ≈ opp_FGA - opp_3PA
    opp_2pa = (df["opp_fieldGoalsAttempted"] - df["opp_threePointersAttempted"]).replace(0, 1)
    df["BLK_PCT"] = df["team_blocks"] / opp_2pa
    df["STL_PCT"] = df["team_steals"] / poss_team

    # ---------- 7. Rebounding ----------
    df["TOTAL_REB_PCT"] = df["team_reboundsTotal"] / (
        df["team_reboundsTotal"] + df["opp_reboundsTotal"]
    ).replace(0, 1)

    # ---------- 8. Scoring distribution ----------
    # points from 2pt = total - FT - 3*3PM
    df["PCT_PTS_2PT"] = (
        df["team_points"]
        - df["team_freeThrowsMade"]
        - 3 * df["team_threePointersMade"]
    ) / df["team_points"].replace(0, 1)

    df["PCT_PTS_3PT"] = (
        3 * df["team_threePointersMade"]
    ) / df["team_points"].replace(0, 1)

    df["PCT_PTS_FT"] = (
        df["team_freeThrowsMade"]
    ) / df["team_points"].replace(0, 1)

    # ---------- 9. Pace / rating diffs ----------
    df["NET_RATING_DIFF"] = df["team_netRating"] - df["opp_netRating"]
    df["PACE_DIFF"] = df["team_pace"] - df["opp_pace"]

    # ---------- 10. Assist / turnover ratios ----------
    denom = (
        df["team_fieldGoalsAttempted"]
        + 0.44 * df["team_freeThrowsAttempted"]
        + df["team_assists"]
        + df["team_turnovers"]
    ).replace(0, 1)

    df["ASSIST_RATIO"] = df["team_assists"] / denom
    df["TURNOVER_RATIO"] = df["team_turnovers"] / denom
#     # Offensive and Defensive Efficiency
#     df['OFF_EFF'] = df['PTS'] / df['POSS']
#     df['DEF_EFF'] = df['PTS_opp'] / df['POSS']

#     # Four Factors differentials
#     df['EFG_DIFF'] = df['EFG_PCT'] - df['EFG_PCT_opp']
#     df['TOV_DIFF'] = df['TM_TOV_PCT'] - df['TM_TOV_PCT_opp']
#     df['OREB_DIFF'] = df['OREB_PCT'] - df['OREB_PCT_opp']
#     df['FT_RATE'] = df['FTA'] / df['FGA']
#     df['FT_RATE_opp'] = df['FTA_opp'] / df['FGA_opp']
#     df['FT_RATE_DIFF'] = df['FT_RATE'] - df['FT_RATE_opp']

#     # Shooting Percentages
#     df['eFG_PCT'] = (df['FGM'] + 0.5 * df['FG3M']) / df['FGA']
#     df['TS_PCT'] = df['PTS'] / (2 * (df['FGA'] + 0.44 * df['FTA']))

#     # Possession-based metrics
#     df['PTS_PER_POSS'] = df['PTS'] / df['POSS']
#     df['AST_PER_POSS'] = df['AST'] / df['POSS']
#     df['TOV_PER_POSS'] = df['TOV'] / df['POSS']

#     # Defensive metrics
#     df['BLK_PCT'] = df['BLK'] / (df['FGA_opp'] - df['FG3A_opp'])
#     df['STL_PCT'] = df['STL'] / df['POSS']

#     # Rebounding metrics
#     df['TOTAL_REB_PCT'] = df['REB'] / (df['REB'] + df['REB_opp'])

#     # Scoring distribution
#     df['PCT_PTS_2PT'] = (df['PTS'] - df['FTM'] - (3 * df['FG3M'])) / df['PTS']
#     df['PCT_PTS_3PT'] = (3 * df['FG3M']) / df['PTS']
#     df['PCT_PTS_FT'] = df['FTM'] / df['PTS']

#     # Pace-adjusted stats
#     df['AST_per_100'] = (df['AST'] / df['POSS']) * 100
#     df['TOV_per_100'] = (df['TOV'] / df['POSS']) * 100

#     # Team comparison metrics
#     df['NET_RATING_DIFF'] = df['NET_RATING'] - df['NET_RATING_opp']
#     df['PACE_DIFF'] = df['PACE'] - df['PACE_opp']

#     # Advanced metrics
#     df['ASSIST_RATIO'] = df['AST'] / (df['FGA'] + 0.44 * df['FTA'] + df['AST'] + df['TOV'])
#     df['TURNOVER_RATIO'] = df['TOV'] / (df['FGA'] + 0.44 * df['FTA'] + df['AST'] + df['TOV'])

    return df

def time_series_engineer(df):
    df.sort_values(by=['GAME_DATE'], inplace=True)
    df['win_streak'] = (df['WL'] == 'W').groupby((df['WL'] != 'W').cumsum()).cumsum()
    df['lose_streak'] = (df['WL'] == 'L').groupby((df['WL'] != 'L').cumsum()).cumsum()
    #Cyclical encoding: For cyclical features like day of week and month, 
    #sine and cosine encoding to capture cyclical nature:
    df['day_of_week'] = df['GAME_DATE'].dt.dayofweek
    df['day_sin'] = np.sin(df['day_of_week'] * (2 * np.pi / 7))
    df['day_cos'] = np.cos(df['day_of_week'] * (2 * np.pi / 7))
    df.drop(columns=['day_of_week'], inplace=True)
    df['pts_diff_last_5'] = df['TEAM_PTS'] - df['TEAM_PTS'].rolling(window=5, min_periods=1).mean()
    # df['days_since_last_game'] = df['GAME_DATE'].diff().dt.days
    df['days_since_last_game'] = (df['GAME_DATE'] - df['GAME_DATE'].shift()).dt.days.fillna(0).astype(int)
    df['AST_TOV_ratio'] = df['TEAM_AST'] / df['TEAM_TOV']
    df['rolling_win_pct'] = (df['WL'] == 'W').astype(int).ewm(alpha=0.3).mean()
    df['rest_3plus'] = (df['days_since_last_game'] >= 3).astype(int)
    df['time_of_game'] = df['team_minutes'].str.split(":").str[0].astype(int)
    df['was_ot'] = (df['time_of_game'] > 240).astype(int)

    # 5) simple opponent-strength signal (only if you have opponent/team id columns)
    if 'TEAM_ID' in df.columns and 'opp_TEAM_ID' in df.columns:
        # rolling win% per team (you already have rolling_win_pct, reuse it)
        # map opponent's rolling win% onto current row
        opp_map = (
            df[['GAME_DATE', 'TEAM_ID', 'rolling_win_pct']]
            .rename(columns={'TEAM_ID': 'opp_TEAM_ID',
                             'rolling_win_pct': 'opp_rolling_win_pct'})
        )
        df = df.merge(
            opp_map[['GAME_DATE', 'opp_TEAM_ID', 'opp_rolling_win_pct']],
            on=['GAME_DATE', 'opp_TEAM_ID'],
            how='left'
        )
        df['opp_rolling_win_pct'] = df['opp_rolling_win_pct'].fillna(0)
        df['win_pct_diff'] = df['rolling_win_pct'] - df['opp_rolling_win_pct']

    return df

def fetch_boxscore_df(
    endpoint_cls,
    game_id,
    *,
    retries=3,
    sleep_after=3,
    backoff=31,
    timeout=31,
):
    """
    Call a v3 boxscore endpoint.
    - sleep_after: sleep every time it succeeds
    - on timeout/conn/JSON error: sleep(backoff) and try again
    - returns empty df if all retries fail
    """
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            resp = endpoint_cls(game_id=game_id, timeout=timeout)

            # v3 sometimes: resp.team_stats, sometimes: resp.get_data_frames()
            if hasattr(resp, "team_stats"):
                df = resp.team_stats.get_data_frame()
            else:
                frames = resp.get_data_frames()
                df = frames[0] if len(frames) else pd.DataFrame()

            time.sleep(sleep_after)
            return df

        except (
            requests.exceptions.ReadTimeout,
            requests.exceptions.ConnectionError,
            ReadTimeoutError,
            json.JSONDecodeError,   # <- the new important one
            ValueError,             # nba_api sometimes wraps bad JSON like this
        ) as e:
            last_err = e
            print(f"[fetch_boxscore_df] {endpoint_cls.__name__} failed for {game_id} "
                  f"(try {attempt}/{retries}): {e}")
            time.sleep(backoff)
            continue

        except Exception as e:
            last_err = e
            print(f"[fetch_boxscore_df] unexpected {endpoint_cls.__name__} for {game_id}: {e}")
            time.sleep(backoff)
            continue

    print(f"[fetch_boxscore_df] giving up on {endpoint_cls.__name__} for {game_id}: {last_err}")
    return pd.DataFrame()

# def fetch_boxscore_df(endpoint_cls, game_id, *, retries=3, sleep_after=3, backoff=31, timeout=31):
#     """
#     Call a v3 boxscore endpoint.
#     - sleep_after: sleep every time it succeeds
#     - on timeout / connection error: sleep(backoff) and try again
#     - returns a *pandas DataFrame* (may be empty if all retries fail)
#     """
#     last_err = None
#     for attempt in range(1, retries + 1):
#         try:
#             # nba_api will forward timeout kwarg
#             resp = endpoint_cls(game_id=game_id, timeout=timeout)
#             # some installs expose .team_stats, others just .get_data_frames()[0]
#             if hasattr(resp, "team_stats"):
#                 df = resp.team_stats.get_data_frame()
#             else:
#                 df = resp.get_data_frames()[0]
#             time.sleep(sleep_after)
#             return df
#         except (requests.exceptions.ReadTimeout,
#                 requests.exceptions.ConnectionError,
#                 ReadTimeoutError) as e:
#             last_err = e
#             print(f"[fetch_boxscore_df] timeout/conn on {endpoint_cls.__name__} for {game_id} "
#                   f"(try {attempt}/{retries}): {e}")
#             time.sleep(backoff)

def collect_data():
    os.makedirs("data", exist_ok=True)

    # if you want to re-run fresh each time, remove this guard

    # seasons that work with BoxScoreAdvancedV2
    seasons = [
        # "2014-15", "2015-16", "2016-17","2017-18", 
         "2018-19", "2019-20","2020-21",
         "2021-22", "2022-23","2023-24",
         "2024-25", "2025-26"
    ]

    all_games = []
    for season in tqdm(seasons, desc="Downloading game lists"):
        gf = LeagueGameFinder(season_nullable=season, league_id_nullable="00")
        df_games = gf.get_data_frames()[0]
        all_games.append(df_games)
        time.sleep(2)

    # every game (two rows per game)
    all_games_df = pd.concat(all_games, ignore_index=True)
    # sort so groupby is stable
    all_games_df = all_games_df.sort_values(["GAME_DATE", "GAME_ID", "TEAM_ID"]).reset_index(drop=True)

    rows = []

    # helper to split team/opp from a boxscore df
    def _split(df_box, team_abbrev):
        team_part = df_box[df_box["teamTricode"] == team_abbrev].copy().add_prefix("team_")
        opp_part = df_box[df_box["teamTricode"] != team_abbrev].copy().add_prefix("opp_")
        return (
            team_part.reset_index(drop=True),
            opp_part.reset_index(drop=True),
        )
    for team in ALL_TEAMS:
        final_df = pd.DataFrame()
        team_df = all_games_df[all_games_df["TEAM_ABBREVIATION"] == team]
        unique_game_ids = team_df["GAME_ID"].unique()
        check = 0
        for game_id in tqdm(unique_game_ids, desc="Fetching boxscores"):
            # two rows: team + opponent
            game_rows = all_games_df[all_games_df["GAME_ID"] == game_id].reset_index(drop=True)
            # in-season tournament / odd games could break this
            if len(game_rows) != 2:
                # skip weird cases
                continue

            # pick row 0 as "our" team, row 1 as opponent
            team_meta = game_rows.loc[0].copy()
            opp_meta = game_rows.loc[1].copy()
            team_abbrev = team_meta["TEAM_ABBREVIATION"]
            opp_abbrev = opp_meta["TEAM_ABBREVIATION"]

            if team_abbrev not in ALL_TEAMS or opp_abbrev not in ALL_TEAMS:
                print(f'{team_abbrev} or {opp_abbrev} is not in the NBA list')
                continue

            if not (game_id.startswith("002") or game_id.startswith("004")):
                #002 = regular season, 004 = playoffs
                continue
            print(f'{team_abbrev} vs. {opp_abbrev}: {game_id}')
            # try:
            adv = fetch_boxscore_df(BoxScoreAdvancedV3,game_id)
            trad = fetch_boxscore_df(BoxScoreTraditionalV3,game_id)
            ff = fetch_boxscore_df(BoxScoreFourFactorsV3,game_id)
            misc = fetch_boxscore_df(BoxScoreMiscV3,game_id)
            score = fetch_boxscore_df(BoxScoreScoringV3,game_id)

            print(score)
            if all([
                not adv.empty,
                not trad.empty,
                not ff.empty,
                not misc.empty,
                not score.empty,
            ]
            ):
                # 2) split each collection into team/opp
                team_adv,  opp_adv  = _split(adv,  team_abbrev)
                team_trad, opp_trad = _split(trad, team_abbrev)
                team_ff,   opp_ff   = _split(ff,   team_abbrev)
                team_misc, opp_misc = _split(misc, team_abbrev)
                team_score, opp_score = _split(score, team_abbrev)

                # 3) build meta pieces
                #    keep GAME_ID, GAME_DATE, MATCHUP, WL, TEAM_ID ...
                team_meta_df = pd.DataFrame([{
                    "SEASON_ID": team_meta["SEASON_ID"],
                    "GAME_ID": game_id,
                    "GAME_DATE": pd.to_datetime(team_meta["GAME_DATE"]),
                    "MATCHUP": team_meta["MATCHUP"],
                    "WL": team_meta["WL"],
                    "TEAM_ID": team_meta["TEAM_ID"],
                    "TEAM_ABBREVIATION": team_meta["TEAM_ABBREVIATION"],
                    "TEAM_NAME": team_meta["TEAM_NAME"],
                    "TEAM_PTS": team_meta['PTS'],
                    "TEAM_MIN":        team_meta["MIN"],
                    "TEAM_FGM":        team_meta["FGM"],
                    "TEAM_FGA":        team_meta["FGA"],
                    "TEAM_FG_PCT":     team_meta["FG_PCT"],
                    "TEAM_FG3M":       team_meta["FG3M"],
                    "TEAM_FG3A":       team_meta["FG3A"],
                    "TEAM_FG3_PCT":    team_meta["FG3_PCT"],
                    "TEAM_FTM":        team_meta["FTM"],
                    "TEAM_FTA":        team_meta["FTA"],
                    "TEAM_FT_PCT":     team_meta["FT_PCT"],
                    "TEAM_OREB":       team_meta["OREB"],
                    "TEAM_DREB":       team_meta["DREB"],
                    "TEAM_REB":        team_meta["REB"],
                    "TEAM_AST":        team_meta["AST"],
                    "TEAM_STL":        team_meta["STL"],
                    "TEAM_BLK":        team_meta["BLK"],
                    "TEAM_TOV":        team_meta["TOV"],
                    "TEAM_PF":         team_meta["PF"],
                    "TEAM_PLUS_MINUS": team_meta["PLUS_MINUS"],
                }])

                opp_meta_df = pd.DataFrame([{
                    "opp_TEAM_ID": opp_meta["TEAM_ID"],
                    "opp_TEAM_ABBREVIATION": opp_meta["TEAM_ABBREVIATION"],
                    "opp_TEAM_NAME": opp_meta["TEAM_NAME"],
                    "opp_WL": opp_meta["WL"],
                    "opp_TEAM_PTS": opp_meta['PTS'],
                    "opp_TEAM_MIN":        opp_meta["MIN"],
                    "opp_TEAM_FGM":        opp_meta["FGM"],
                    "opp_TEAM_FGA":        opp_meta["FGA"],
                    "opp_TEAM_FG_PCT":     opp_meta["FG_PCT"],
                    "opp_TEAM_FG3M":       opp_meta["FG3M"],
                    "opp_TEAM_FG3A":       opp_meta["FG3A"],
                    "opp_TEAM_FG3_PCT":    opp_meta["FG3_PCT"],
                    "opp_TEAM_FTM":        opp_meta["FTM"],
                    "opp_TEAM_FTA":        opp_meta["FTA"],
                    "opp_TEAM_FT_PCT":     opp_meta["FT_PCT"],
                    "opp_TEAM_OREB":       opp_meta["OREB"],
                    "opp_TEAM_DREB":       opp_meta["DREB"],
                    "opp_TEAM_REB":        opp_meta["REB"],
                    "opp_TEAM_AST":        opp_meta["AST"],
                    "opp_TEAM_STL":        opp_meta["STL"],
                    "opp_TEAM_BLK":        opp_meta["BLK"],
                    "opp_TEAM_TOV":        opp_meta["TOV"],
                    "opp_TEAM_PF":         opp_meta["PF"],
                    "opp_TEAM_PLUS_MINUS": opp_meta["PLUS_MINUS"],
                }])

                # 4) concat everything horizontally
                final_row = pd.concat(
                    [
                        team_meta_df.reset_index(drop=True),
                        opp_meta_df.reset_index(drop=True),
                        team_trad.reset_index(drop=True),
                        opp_trad.reset_index(drop=True),
                        team_adv.reset_index(drop=True),
                        opp_adv.reset_index(drop=True),
                        team_ff.reset_index(drop=True),
                        opp_ff.reset_index(drop=True),
                        team_misc.reset_index(drop=True),
                        opp_misc.reset_index(drop=True),
                        team_score.reset_index(drop=True),
                        opp_score.reset_index(drop=True),
                    ],
                    axis=1,
                )
                final_row = final_row.loc[:, ~final_row.columns.duplicated()].copy()
                rows.append(final_row)
                # check += 1
                # if check == 3:
                #     break
            else:
                print(f'[SKIP] skipping row due to inability to collect game')

                # except Exception as e:
                #     print(f"[warn] error for game {game_id}: {e}")
                #     continue

                # stack all games
            final_df = pd.concat(rows, ignore_index=True)
            time.sleep(2)

            # ----- feature engineering -----
        # make sure GAME_DATE is datetime
        final_df["GAME_DATE"] = pd.to_datetime(final_df["GAME_DATE"])

        # your existing feature logic
        final_df = engineer_features(final_df)

        # time-series / rest / streak logic
        final_df = time_series_engineer(final_df)

        #calendar features
        final_df = _add_calendar_features(final_df)

        # save
        out_path = f"data/{team}_all_games.csv"
        final_df.to_csv(out_path, index=False)
    return final_df
