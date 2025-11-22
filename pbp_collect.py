import os
import time
import pandas as pd
from tqdm import tqdm
from nba_api.stats.endpoints import LeagueGameFinder, PlayByPlayV3
from nba_api.live.nba.endpoints import scoreboard as live_scoreboard
import re
import math
import pandas as pd  # you’re already using this
import time
from requests.exceptions import ReadTimeout, RequestException 
import random

# seasons you care about – match what you're already using
SEASONS = [
    "2018-19", "2019-20", "2020-21",
    "2021-22", "2022-23", "2023-24",
    "2024-25", "2025-26"
]

def _build_game_meta_this_season(seasons=["2025-26"]):
    """
    Get one row per GAME_ID with:
    - home/away teams
    - date
    - final score
    - who won
    """
    all_games = []
    for season in seasons:
        gf = LeagueGameFinder(
            season_nullable=season,
            league_id_nullable="00",
            team_id_nullable=None,
        )
        df = gf.get_data_frames()[0]
        df["SEASON"] = season
        all_games.append(df)
        time.sleep(0.6)

    games = pd.concat(all_games, ignore_index=True)
    meta_rows = []
    for game_id, g in games.groupby("GAME_ID"):
        # regular games should have exactly 2 rows (one per team)
        if len(g) != 2:
            continue

        g = g.sort_values("TEAM_ID").reset_index(drop=True)
        row1 = g.iloc[0]
        row2 = g.iloc[1]

        matchup = row1["MATCHUP"]

        # "GSW vs. LAL" -> row1 is home
        # "GSW @ LAL"   -> row1 is away, row2 is home
        if "@ " in matchup or "@" in matchup:
            away = row1
            home = row2
        else:
            home = row1
            away = row2

        meta_rows.append(
            {
                "SEASON": row1["SEASON"],
                "GAME_ID": game_id,
                "GAME_DATE": pd.to_datetime(home["GAME_DATE"]),
                "HOME_TEAM_ABBREV": home["TEAM_ABBREVIATION"],
                "AWAY_TEAM_ABBREV": away["TEAM_ABBREVIATION"],
                "HOME_TEAM_ID": home["TEAM_ID"],
                "AWAY_TEAM_ID": away["TEAM_ID"],
                "HOME_FINAL_PTS": home["PTS"],
                "AWAY_FINAL_PTS": away["PTS"],
                "HOME_WIN": 1 if home["WL"] == "W" else 0,
            }
        )

    return pd.DataFrame(meta_rows)

def build_today_live_meta():
    sb = live_scoreboard.ScoreBoard()  # defaults to "today" (UTC)
    data = sb.get_dict()["scoreboard"]["games"]
    rows = []
    for g in data:
        rows.append({
            # "SEASON": sb.get_dict()["scoreboard"]["season"],
            "GAME_ID": g["gameId"],
            "GAME_DATE": pd.to_datetime(g["gameEt"]),     # convert as you like
            "HOME_TEAM_ABBREV": g["homeTeam"]["teamTricode"],
            "AWAY_TEAM_ABBREV": g["awayTeam"]["teamTricode"],
            "HOME_TEAM_ID": g["homeTeam"]["teamId"],
            "AWAY_TEAM_ID": g["awayTeam"]["teamId"],
            "HOME_PTS": int(g["homeTeam"]["score"] or 0),
            "AWAY_PTS": int(g["awayTeam"]["score"] or 0),
            "STATUS": g["gameStatus"],           # 1=Scheduled, 2=In-Progress, 3=Final
            "STATUS_TEXT": g["gameStatusText"],  # e.g., "1Q 5:00", "Final"
        })
    return pd.DataFrame(rows)

def _build_game_meta(seasons=SEASONS):
    """
    Get one row per GAME_ID with:
    - home/away teams
    - date
    - final score
    - who won
    """
    all_games = []
    for season in seasons:
        gf = LeagueGameFinder(
            season_nullable=season,
            league_id_nullable="00",
            team_id_nullable=None,
        )
        df = gf.get_data_frames()[0]
        df["SEASON"] = season
        all_games.append(df)
        time.sleep(0.6)

    games = pd.concat(all_games, ignore_index=True)

    meta_rows = []
    for game_id, g in games.groupby("GAME_ID"):
        # regular games should have exactly 2 rows (one per team)
        if len(g) != 2:
            continue

        g = g.sort_values("TEAM_ID").reset_index(drop=True)
        row1 = g.iloc[0]
        row2 = g.iloc[1]

        matchup = row1["MATCHUP"]

        # "GSW vs. LAL" -> row1 is home
        # "GSW @ LAL"   -> row1 is away, row2 is home
        if "@ " in matchup or "@" in matchup:
            away = row1
            home = row2
        else:
            home = row1
            away = row2

        meta_rows.append(
            {
                "SEASON": row1["SEASON"],
                "GAME_ID": game_id,
                "GAME_DATE": pd.to_datetime(home["GAME_DATE"]),
                "HOME_TEAM_ABBREV": home["TEAM_ABBREVIATION"],
                "AWAY_TEAM_ABBREV": away["TEAM_ABBREVIATION"],
                "HOME_TEAM_ID": home["TEAM_ID"],
                "AWAY_TEAM_ID": away["TEAM_ID"],
                "HOME_FINAL_PTS": home["PTS"],
                "AWAY_FINAL_PTS": away["PTS"],
                "HOME_WIN": 1 if home["WL"] == "W" else 0,
            }
        )

    return pd.DataFrame(meta_rows)


def _parse_time(period, pctimestring):
    """
    Convert (period, 'MM:SS' time left in period) to:
      - seconds from game start
      - seconds remaining in regulation (48 min; OT clamped at 0)
    """
    if not isinstance(pctimestring, str) or pctimestring.strip() == "":
        return None, None

    mm, ss = pctimestring.split(":")
    mm = int(mm)
    ss = int(ss)
    sec_left_in_period = mm * 60 + ss

    if period <= 4:
        # 12-min quarters
        sec_from_start = (period - 1) * 12 * 60 + (12 * 60 - sec_left_in_period)
    else:
        # 5-min OT periods
        sec_from_start = 48 * 60 + (period - 5) * 5 * 60 + (5 * 60 - sec_left_in_period)

    sec_remaining_reg = max(0, 48 * 60 - sec_from_start)
    return sec_from_start, sec_remaining_reg

def _parse_clock_to_seconds_left(clock_str):
    """
    clock_str: either '11:23' or 'PT11M23.00S' etc.
    returns: seconds remaining in the *period* (int) or None
    """
    if not isinstance(clock_str, str):
        return None
    s = clock_str.strip()
    if not s:
        return None

    # Simple MM:SS
    if ":" in s and not s.startswith("PT"):
        mm, ss = s.split(":")
        try:
            return int(mm) * 60 + int(float(ss))
        except ValueError:
            return None

    # ISO-ish: PTmmMss.ccS
    if s.startswith("PT"):
        m = re.match(r"PT(\d+)M(\d+(?:\.\d+)?)S", s)
        if not m:
            return None
        mm = int(m.group(1))
        ss = float(m.group(2))
        return int(round(mm * 60 + ss))

    return None


def _parse_time_v3(period, clock_str):
    """
    period: 1,2,3,4,... (OT periods are 5+)
    clock_str: whatever 'clock' column holds

    returns:
       SECONDS_FROM_START (monotonic up),
       SECONDS_REMAINING_REG (0..48*60, clamped at 0 in OT)
    """
    sec_left_in_period = _parse_clock_to_seconds_left(clock_str)
    if sec_left_in_period is None:
        return None, None

    REG_Q_LEN = 12 * 60
    OT_LEN = 5 * 60

    if period <= 4:
        sec_from_start = (period - 1) * REG_Q_LEN + (REG_Q_LEN - sec_left_in_period)
    else:
        # everything after 48 mins
        sec_from_start = 48 * 60 + (period - 5) * OT_LEN + (OT_LEN - sec_left_in_period)

    sec_remaining_reg = max(0, 48 * 60 - sec_from_start)
    return sec_from_start, sec_remaining_reg

def _get_pbp_with_retry(game_id, max_retries=5, sleep_secs=30):
    """
    Try to fetch PlayByPlayV3 up to max_retries times.
    On ReadTimeout, wait sleep_secs and retry.
    After max_retries failures, return None so caller can skip this game.
    """
    for attempt in range(1, max_retries + 1):
        try:
            pbp = PlayByPlayV3(game_id=game_id, timeout=30).get_data_frames()[0]
            return pbp
        except ReadTimeout as e:
            print(
                f"[WARN] PlayByPlayV3 timeout for game {game_id} "
                f"(attempt {attempt}/{max_retries}): {e}"
            )
            if attempt == max_retries:
                print(f"[WARN] Giving up on game {game_id} after {max_retries} timeouts.")
                return None
            time.sleep(sleep_secs)
        except RequestException as e:
            # Other network errors – log and give up on this game
            print(f"[WARN] HTTP error for game {game_id}: {e}")
            return None
        except Exception as e:
            # Anything else unexpected – log and give up on this game
            print(f"[WARN] Unexpected error for game {game_id}: {e}")
            return None

def _expand_pbp_for_game(game_row):
    """
    For a single game, pull PlayByPlayV2 and turn it into
    one row per event with:
      - time features
      - score features
      - text descriptions (for later NLP if you want)
    """
    game_id = game_row["GAME_ID"]

    print(game_id)

    # pbp = PlayByPlayV3(game_id=game_id, timeout=30).get_data_frames()[0]
    pbp = _get_pbp_with_retry(game_id)
    if pbp is None or pbp.empty:
        return None
    # print(pbp.columns.tolist())

    pbp = pbp.sort_values(["period", "actionNumber"]).reset_index(drop=True)

    pbp["scoreHome"] = pd.to_numeric(pbp["scoreHome"], errors="coerce")
    pbp["scoreAway"] = pd.to_numeric(pbp["scoreAway"], errors="coerce")

    pbp[["scoreHome", "scoreAway"]] = pbp[["scoreHome", "scoreAway"]].ffill()
    pbp[["scoreHome", "scoreAway"]] = pbp[["scoreHome", "scoreAway"]].fillna(0)

    rows = []
    for _, ev in pbp.iterrows():
        period = int(ev["period"])
        clock_str = ev["clock"]

        sec_from_start, sec_remaining_reg = _parse_time_v3(period, clock_str)

        home_pts = ev.get("scoreHome")
        away_pts = ev.get("scoreAway")
        lead_home = int(home_pts) - int(away_pts)
        rows.append(
            {
                "SEASON": game_row["SEASON"],
                "GAME_ID": game_id,
                "GAME_DATE": game_row["GAME_DATE"],
                "HOME_TEAM_ABBREV": game_row["HOME_TEAM_ABBREV"],
                "AWAY_TEAM_ABBREV": game_row["AWAY_TEAM_ABBREV"],
                "HOME_WIN": game_row["HOME_WIN"],

                "period": period,
                "actionNumber": ev["actionNumber"],
                "clock": clock_str,
                "SECONDS_FROM_START": sec_from_start,
                "SECONDS_REMAINING_REG": sec_remaining_reg,

                "HOME_PTS": home_pts,
                "AWAY_PTS": away_pts,
                "LEAD_HOME": lead_home,

                "teamTricode": ev["teamTricode"],
                "description": ev["description"],
                "actionType": ev["actionType"],
                "subType": ev["subType"],
            }
        )
    return pd.DataFrame(rows)

def _save_ckpt(i_next, ckpt_path):
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    with open(ckpt_path, "w") as f:
        f.write(str(i_next))

def collect_pbp_states(
    seasons=SEASONS,
    out_path="data/pbp_states.csv",
):
    """
    Main entrypoint:
      - builds per-game meta (home/away, final outcome)
      - loops over games and expands PlayByPlay into 'state' rows
      - saves one big CSV with every event of every game
    """
    os.makedirs("data", exist_ok=True)

    game_meta = _build_game_meta(seasons)
    print(f"Found {len(game_meta)} games across {len(seasons)} seasons")
    all_states = []
    ckpt_path = "data/pbp_checkpoint.txt"
    if os.path.exists(out_path):
        final_df = pd.read_csv(out_path)
    else:
        final_df = None
    write_header = not os.path.exists(out_path) or os.path.getsize(out_path) == 0
    start_idx = 0
    if os.path.exists(ckpt_path):
        try:
            with open(ckpt_path, "r") as f:
                start_idx = int(f.read().strip() or "0")
        except:
            start_idx = 0
    for i, row in tqdm(game_meta.iterrows(), total=len(game_meta), desc="PBP"):
        if i < start_idx:
            continue  # already processed in prior runs
        df_states = _expand_pbp_for_game(row)
        print(df_states)
        if df_states is not None and not df_states.empty:
            df_states.to_csv(out_path, mode="a", header=write_header, index=False)
        write_header = False
        _save_ckpt(i + 1,ckpt_path)
        time.sleep(random.uniform(4, 10))
        # print(df_states)
        # if df_states is not None:
        #     all_states.append(df_states)
            
        #     if final_df == None:
        #         final_df = pd.concat(all_states, ignore_index=True)
        #         final_df.to_csv(out_path, index=False)
        #     else:
        #         final_df = pd.concat([final_df, all_states], ignore_index=True)
        # time.sleep(random.randint(5, 10))  # be nice to the API


    print(f"Saved {len(final_df)} play-by-play rows -> {out_path}")
    return final_df
