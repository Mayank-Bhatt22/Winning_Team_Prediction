# Data generator module for creating synthetic IPL match data

import pandas as pd
import numpy as np
import random


# Reproducibility
np.random.seed(42)
random.seed(42)


# IPL Teams
TEAMS = [
    'MI', 'CSK', 'RCB', 'KKR', 'SRH',
    'DC', 'PBKS', 'RR', 'GT', 'LSG'
]


# IPL Venues
VENUES = [
    'Wankhede Stadium',
    'M. A. Chidambaram Stadium',
    'M. Chinnaswamy Stadium',
    'Eden Gardens',
    'Feroz Shah Kotla Ground',
    'Punjab Cricket Association IS Bindra Stadium',
    'Sawai Mansingh Stadium',
    'Narendra Modi Stadium',
    'Brabourne Stadium',
    'Rajiv Gandhi International Cricket Stadium'
]


def generate_match_data(num_matches=1000):

    rows = []

    for i in range(num_matches):

        # Select two different teams
        team1 = random.choice(TEAMS)
        team2 = random.choice(
            [team for team in TEAMS if team != team1]
        )

        # Match information
        venue = random.choice(VENUES)

        toss_winner = random.choice([team1, team2])

        toss_decision = random.choice(['bat', 'field'])

        # Recent form:
        # Average runs scored in the previous 5 matches
        team1_runs_last_5 = random.randint(130, 210)
        team2_runs_last_5 = random.randint(130, 210)

        # Team strength
        team1_strength = random.uniform(0.35, 0.75)
        team2_strength = random.uniform(0.35, 0.75)

        # Recent form converted into a small advantage
        team1_form = (team1_runs_last_5 - 170) / 500
        team2_form = (team2_runs_last_5 - 170) / 500

        # Base strength
        team1_score = team1_strength + team1_form
        team2_score = team2_strength + team2_form

        # Toss advantage
        if toss_winner == team1:
            team1_score += 0.08
        else:
            team2_score += 0.08

        # Small random factor so that stronger team doesn't
        # always win
        team1_score += random.uniform(-0.08, 0.08)
        team2_score += random.uniform(-0.08, 0.08)

        # Decide winner
        if team1_score > team2_score:
            winner = team1
        else:
            winner = team2

        # Store match
        rows.append({
            'team1': team1,
            'team2': team2,
            'venue': venue,
            'toss_winner': toss_winner,
            'toss_decision': toss_decision,
            'team1_runs_last_5': team1_runs_last_5,
            'team2_runs_last_5': team2_runs_last_5,
            'winner': winner
        })

    # Convert to DataFrame
    df = pd.DataFrame(rows)

    # Save dataset
    df.to_csv(
        'synthetic_match_data.csv',
        index=False
    )

    print(
        f"{len(df)} synthetic match data generated "
        "and saved to 'synthetic_match_data.csv'."
    )

    return df


# Run the generator
if __name__ == "__main__":
    generate_match_data(num_matches=1500)
