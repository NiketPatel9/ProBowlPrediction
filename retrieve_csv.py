"""
This module is responsible for extracting CSVs as well as cleaning them up and combining them.
"""
import requests
import pandas as pd
from bs4 import BeautifulSoup


def create_csvs():
    """
    This function is responsible for web scraping various years from the Pro Football Database as well
    as cleaning up the CSVs and making the data relevant.

    :return: void
    """

    dfs = []

    # 1960 is the first year of the AFL-NFL merger, and we will exclude 2020
    for year in range(1960, 2020):

        # Web scraping is accomplished here with the requests and BeautifulSoup modules
        pfr_url = f"https://www.pro-football-reference.com/years/{year}/passing.htm"
        page = requests.get(pfr_url)

        soup = BeautifulSoup(page.content, 'html.parser')
        results = soup.find(id='passing')

        df = pd.read_html(results.prettify())[0]

        # drop columns that are not consistent throughout history
        df = df.drop(['1D', 'QBR'], axis=1, errors='ignore')

        # Remove rows that are not players
        df = df[df.Att != "Att"]

        # For our purposes, QBs are any players that have a QB record.
        df = df[df['QBrec'].notna()]

        dfs.append(df)

    # Combine all years, drop columns that are not consistent
    full_df = pd.concat(dfs)
    full_df = full_df.drop(['Rk', 'Tm', 'Pos', 'Sk', 'Yds.1', 'NY/A', 'ANY/A', 'Sk%'], axis=1, errors='ignore')
    full_df.to_csv(f"passing_old.csv", index=False)


def rename_csv_cols():
    """
    This function reads in the CSV created from the create_csvs() function and renames columns to more
    friendly names, creates a few new columns.
    :return: void
    """

    # Creates labels for Pro Bowler vs non-Pro Bowler as well as some basic feature engineering
    # to create win, loss, and tie columns.
    passing = pd.read_csv("passing_old.csv")
    passing = passing.fillna(0)
    passing.loc[~passing['Player'].str.contains('*', regex=False), 'Player'] = '0'
    passing.loc[passing['Player'].str.contains('*', regex=False), 'Player'] = '1'
    passing[['Win', 'Loss', 'Tie']] = passing['QBrec'].str.split('-', expand=True)

    # Dropping and renaming columns for more readability
    passing = passing.drop(["QBrec"], axis=1)
    passing = passing.rename(columns={"Player": "pro_bowl", "G": "games_played", "GS": "games_started",
                                      "Cmp": "completions", "Att": "attempts", "Cmp%": 'completion_percent',
                                      "Yds": "yards", "TD": "touchdowns", "TD%": "touchdown_percent",
                                      "Int": "interceptions", "Int%": "interception_percentage",
                                      "Lng": "longest_pass", "Y/A": "yards_per_attempt",
                                      "AY/A": "adjusted_yards_per_attempt", "Y/C": "yards_per_completion",
                                      "Y/G": "yards_per_game", "Rate": "qb_rating",
                                      "4QC": "fourth_quarter_comebacks", "GWD": "game_winning_drives"})

    passing.to_csv(f"passing.csv", index=False)


if __name__ == "__main__":
    create_csvs()
    rename_csv_cols()
