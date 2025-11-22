import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm
import os
ALL_TEAMS = ['DAL','PHI', 'BOS', 'CHA', 'POR', 'TOR', 'GSW', 'MIN', 'UTA',
 'MIA', 'SAS', 'NOP', 'CHI', 'DEN', 'LAL', 'LAC', 'ORL', 'NYK', 'WAS', 'ATL',
 'PHX', 'SAC', 'CLE', 'HOU', 'MEM', 'BKN', 'MIL', 'OKC', 'DET', 'IND'
 ]
def read_all_betting_csvs():
    all_csvs = {}
    for one, two in zip(np.arange(2018,2026),np.arange(2019,2027)):
        betting_data_path = f'data/{int(one)}-{int(two)}.csv'
        all_csvs[(one, two)] = pd.read_csv(betting_data_path)
    return all_csvs

def convert_date(date_str):
    yesterday = datetime.now().date() - timedelta(days=1)
    if date_str.startswith("Yesterday"):
        return yesterday.strftime("%d %b %Y")
    else:
        #it's not "Yesterday" - original string
        return date_str
    
def preprocess():
    """
    -one hot encode the actionType features
    """
    name_to_abbr = {
        "Atlanta Hawks": "ATL",
        "Boston Celtics": "BOS",
        "Brooklyn Nets": "BKN",
        "Charlotte Hornets": "CHA",
        "Chicago Bulls": "CHI",
        "Cleveland Cavaliers": "CLE",
        "Dallas Mavericks": "DAL",
        "Denver Nuggets": "DEN",
        "Detroit Pistons": "DET",
        "Golden State Warriors": "GSW",
        "Houston Rockets": "HOU",
        "Indiana Pacers": "IND",
        "Los Angeles Clippers": "LAC",
        "Los Angeles Lakers": "LAL",
        "Memphis Grizzlies": "MEM",
        "Miami Heat": "MIA",
        "Milwaukee Bucks": "MIL",
        "Minnesota Timberwolves": "MIN",
        "New Orleans Pelicans": "NOP",
        "New York Knicks": "NYK",
        "Oklahoma City Thunder": "OKC",
        "Orlando Magic": "ORL",
        "Philadelphia 76ers": "PHI",
        "Phoenix Suns": "PHX",
        "Portland Trail Blazers": "POR",
        "Sacramento Kings": "SAC",
        "San Antonio Spurs": "SAS",
        "Toronto Raptors": "TOR",
        "Utah Jazz": "UTA",
        "Washington Wizards": "WAS",
    }
    df_pbp = pd.read_csv("data/pbp_states.csv")
    all_bet_dfs = read_all_betting_csvs()

    for team_name in tqdm(sorted(df_pbp['HOME_TEAM_ABBREV'].unique())):
        team_df = df_pbp[df_pbp['HOME_TEAM_ABBREV'] == team_name].dropna()
        if team_name not in ALL_TEAMS:
            print((f'Skipping {team_name}'))
            continue
        print(f'Processing {team_name}')
        team_df_save = pd.DataFrame()
        print(f'Total games for {team_name}: {team_df.shape[0]}')
        for val, row in tqdm(team_df.iterrows()):
            if (str(row['HOME_TEAM_ABBREV']) not in ALL_TEAMS) or (str(row['AWAY_TEAM_ABBREV']) not in ALL_TEAMS):
                # print(f"Skip {row['HOME_TEAM_ABBREV']} vs {row['AWAY_TEAM_ABBREV']}")
                continue
            first_year = int(row['SEASON'][0:4])
            second_year = int("20" + row['SEASON'][5:8])

            season_df_bet = pd.DataFrame()
            season_df_bet = all_bet_dfs[(first_year, second_year)].copy()
            season_df_bet['Date'] = season_df_bet['Date'].apply(convert_date)
            season_df_bet['Date'] = pd.to_datetime(season_df_bet['Date'].str.split('-').str[0].str.strip(), format='%d %b %Y')
            season_df_bet["Team_abbr"] = season_df_bet["Team"].map(name_to_abbr)
            season_df_bet["Opp_abbr"] = season_df_bet["Opposition"].map(name_to_abbr)
            
            game_data = season_df_bet.loc[
                (season_df_bet['Date'] == row["GAME_DATE"]) &
                ((season_df_bet['Team_abbr'] == row['HOME_TEAM_ABBREV']) | (season_df_bet['Opp_abbr'] == row['AWAY_TEAM_ABBREV']))
            ]

            combined_df = pd.concat([row.to_frame().T.reset_index(drop=True), game_data.reset_index(drop=True)], axis=1)
            team_df_save = pd.concat([team_df_save, combined_df], ignore_index=True)
        os.makedirs('betting_data',exist_ok=True)
        team_df_save.to_csv(f'betting_data/{team_name}_betting_data.csv', index=False)    