import pandas as pd

def compute_elo(df, k=20, base_elo=1500):

    teams = pd.concat([df["HomeTeam"], df["AwayTeam"]]).unique()
    elo = {team: base_elo for team in teams}

    home_elos = []
    away_elos = []

    for _, row in df.iterrows():

        home = row["HomeTeam"]
        away = row["AwayTeam"]

        home_elo = elo[home]
        away_elo = elo[away]

        home_elos.append(home_elo)
        away_elos.append(away_elo)

        expected_home = 1 / (1 + 10 ** ((away_elo - home_elo) / 400))
        expected_away = 1 - expected_home

        if row["FTR"] == "H":
            s_home, s_away = 1, 0
        elif row["FTR"] == "A":
            s_home, s_away = 0, 1
        else:
            s_home, s_away = 0.5, 0.5

        elo[home] = home_elo + k * (s_home - expected_home)
        elo[away] = away_elo + k * (s_away - expected_away)

    df["home_elo"] = home_elos
    df["away_elo"] = away_elos
    df["elo_diff"] = df["home_elo"] - df["away_elo"]

    return df