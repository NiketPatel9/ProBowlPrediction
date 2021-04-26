"""
This module reads in the three saved models from the full experiment and performs predictions based on an
ensemble classifier made up of those models. Please create the CSV file beforehand with the following columns:
Age, games_played, games_started, completions, attempts, completion_percent, yards, touchdowns, touchdown_percent,
interceptions, interception_percentage, longest_pass, yards_per_attempt, adjusted_yards_per_attempt,
yards_per_completion, yards_per_game, qb_rating, fourth_quarter_comebacks, game_winning_drives, Win, Loss
"""
import pickle
from keras.models import load_model
from preprocess import standardize
import pandas as pd
import numpy as np
from scipy import stats


def predict_players(players):
    """
    This function accepts a dataframe of player stats that are relevant in this experiment, preprocesses those
    stats, and then makes an ensemble prediction on whether those players would make the Pro Bowl.
    :param players: A dataframe of players stats.
    :return: void
    """
    players = standardize(players)

    rf = pickle.load(open('rf.sav', 'rb'))
    lrm = pickle.load(open('lrm.sav', 'rb'))
    nn = load_model('neural_net.h5')

    pred1 = rf.predict(players)
    pred2 = lrm.predict(players)
    pred3 = nn.predict(players)

    for i in range(0, len(pred1)):
        final_pred = stats.mode([pred1[i], pred2[i], pred3[i][0]])[0]
        if final_pred == 0:
            print(f"Player {i + 1} is not a Pro Bowl Quarterback")
        else:
            print(f"Player {i + 1} is a Pro Bowl Quarterback")


if __name__ == "__main__":

    # Please set the CSV path here for now
    players = pd.read_csv("new_players.csv")

    predict_players(players)

