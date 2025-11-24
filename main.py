from odds_hist import get_hist_odds
from pbp_collect import collect_pbp_states
from pbp_preprocess import preprocess
from machine_learning import train_models, predict_win_probability
from backtest import backtest_games
def main():
    #TODO: add command line args to select which steps to run
    # preprocess()
    # collect_pbp_states()
    # get_hist_odds()
    model = train_models()
    # backtest_games(model)
    predict_win_probability(model)

if __name__ == "__main__":
    main()