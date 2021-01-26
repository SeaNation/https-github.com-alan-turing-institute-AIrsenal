"""
Save fitted team models with and without FIFA ratings as covariates for matches up
to each gameweek across all seasons in database.

NOTE: Can take about an hour to run (fits the team model up to 300 times!)
"""

import pickle

from airsenal.framework.utils import get_past_seasons, NEXT_GAMEWEEK, session
from airsenal.framework.season import CURRENT_SEASON, get_teams_for_season
from airsenal.framework.bpl_interface import (
    get_fitted_team_model,
    create_and_fit_team_model,
    get_result_df,
)


def fit_nofifa(season, gameweek, session):
    """Fit team model without using FIFA ratings as covariates."""
    df_team = get_result_df(season, gameweek, session)
    teams = get_teams_for_season(season, session)
    model_nofifa = create_and_fit_team_model(df_team, None, teams=teams)
    return model_nofifa


def get_valid_gameweeks(season):
    if season == "1920":
        # Covid-19 disruption - no gameweek 30-38, gameweek 39-47 instead
        gameweeks = list(range(1, 30)) + list(range(39, 48))

    elif season == CURRENT_SEASON:
        # only fit up to when we have results
        gameweeks = range(1, NEXT_GAMEWEEK)

    elif season == min(seasons):
        # oldest season in database - need to have at least 1 round of results to fit
        # model so start from gameweek 2
        gameweeks = range(2, 39)

    else:
        # whole season
        gameweeks = range(1, 39)

    return gameweeks


def save_models(season):
    gameweeks = get_valid_gameweeks(season)

    models_fifa = {}
    models_nofifa = {}
    for gameweek in gameweeks:
        print(f"GW {gameweek}")
        try:
            models_fifa[gameweek] = get_fitted_team_model(season, gameweek, session)
        except RuntimeError:
            # if initialisation failed try again (just once)
            models_fifa[gameweek] = get_fitted_team_model(season, gameweek, session)
        try:
            models_nofifa[gameweek] = fit_nofifa(season, gameweek, session)
        except RuntimeError:
            # if initialisation failed try again (just once)
            models_nofifa[gameweek] = fit_nofifa(season, gameweek, session)

    print(f"Done fitting models for {season}")

    fifa_path = f"data/models_fifa_{season}.pkl"
    with open(fifa_path, "wb") as f:
        pickle.dump(models_fifa, f)
    print("Saved file:", fifa_path)

    nofifa_path = f"data/models_nofifa_{season}.pkl"
    with open(nofifa_path, "wb") as f:
        pickle.dump(models_nofifa, f)
    print("Saved file:", nofifa_path)


if __name__ == "__main__":
    seasons = [CURRENT_SEASON] + get_past_seasons(3)
    for season in seasons:
        print(f"----- {season} -----")
        save_models(season)
