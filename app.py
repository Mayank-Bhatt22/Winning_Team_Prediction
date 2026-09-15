import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

from tensorflow.keras.models import load_model

# Import our data generator
from data_generator import generate_match_data, TEAMS, VENUES


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="IPL Match Predictor",
    page_icon="🏏",
    layout="wide"
)


# =========================================================
# LOAD MODEL AND PREPROCESSING FILES
# =========================================================

@st.cache_resource
def load_prediction_model():

    model = load_model("ipl_winner_model.keras")

    with open("label_encoders.pkl", "rb") as f:
        label_encoders = pickle.load(f)

    with open("winner_encoder.pkl", "rb") as f:
        winner_encoder = pickle.load(f)

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    return model, label_encoders, winner_encoder, scaler


model, le_dict, le_winner, scaler = load_prediction_model()


# =========================================================
# LOAD DATASET
# =========================================================

@st.cache_data
def load_data():

    if os.path.exists("synthetic_match_data.csv"):
        return pd.read_csv("synthetic_match_data.csv")

    return pd.DataFrame()


df = load_data()


# =========================================================
# SIDEBAR NAVIGATION
# =========================================================

st.sidebar.title("🏏 IPL Predictor")

page = st.sidebar.radio(
    "Go to",
    [
        "🏠 Home",
        "📊 Data Generator",
        "🔮 Prediction",
        "📈 About Teams",
        "ℹ️ About"
    ]
)


# =========================================================
# HOME PAGE
# =========================================================

if page == "🏠 Home":

    st.title("🏏 IPL Match Winner Predictor")

    st.subheader("Predict the winner using Machine Learning")

    st.write(
        """
        This project uses an Artificial Neural Network (ANN) to predict
        the possible winner of an IPL match.
        """
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("🏏 IPL Teams", len(TEAMS))

    with col2:
        if not df.empty:
            st.metric("📊 Matches", len(df))
        else:
            st.metric("📊 Matches", "0")

    with col3:
        st.metric("🧠 Model", "ANN")

    st.divider()

    st.subheader("How does it work?")

    st.write(
        """
        The model looks at information such as:
        
        • Team 1  
        • Team 2  
        • Venue  
        • Toss winner  
        • Toss decision  
        • Team 1 recent form  
        • Team 2 recent form  
        
        It then predicts which team is more likely to win.
        """
    )

    st.info(
        "⚠️ This is a learning/demo project using synthetic data. "
        "The prediction should not be treated as a real IPL prediction."
    )


# =========================================================
# DATA GENERATOR PAGE
# =========================================================

elif page == "📊 Data Generator":

    st.title("📊 Synthetic Match Data Generator")

    st.write(
        "Generate synthetic IPL match data for training and testing."
    )

    num_matches = st.slider(
        "Number of matches",
        min_value=100,
        max_value=10000,
        value=1500,
        step=100
    )

    if st.button("🚀 Generate Data"):

        generated_df = generate_match_data(num_matches)

        st.cache_data.clear()

        st.success(
            f"{num_matches} matches generated successfully!"
        )

        st.dataframe(
            generated_df.head(20),
            use_container_width=True
        )

        csv = generated_df.to_csv(index=False)

        st.download_button(
            label="⬇️ Download CSV",
            data=csv,
            file_name="synthetic_match_data.csv",
            mime="text/csv"
        )


# =========================================================
# PREDICTION PAGE
# =========================================================

elif page == "🔮 Prediction":

    st.title("🔮 IPL Match Prediction")

    st.write(
        "Enter the match details and let the ANN predict the winner."
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        team1 = st.selectbox(
            "Team 1",
            TEAMS
        )

        venue = st.selectbox(
            "Venue",
            VENUES
        )

        team1_runs = st.slider(
            "Team 1 - Average Runs in Last 5 Matches",
            min_value=130,
            max_value=210,
            value=170
        )

    with col2:

        available_team2 = [
            team for team in TEAMS
            if team != team1
        ]

        team2 = st.selectbox(
            "Team 2",
            available_team2
        )

        toss_winner = st.selectbox(
            "Toss Winner",
            [team1, team2]
        )

        team2_runs = st.slider(
            "Team 2 - Average Runs in Last 5 Matches",
            min_value=130,
            max_value=210,
            value=170
        )

    toss_decision = st.selectbox(
        "Toss Decision",
        ["bat", "field"]
    )

    st.divider()

    if st.button("🔮 Predict Winner", use_container_width=True):

        # Create new match dataframe
        new_match = {
            "team1": team1,
            "team2": team2,
            "venue": venue,
            "toss_winner": toss_winner,
            "toss_decision": toss_decision,
            "team1_runs_last_5": team1_runs,
            "team2_runs_last_5": team2_runs
        }

        new_df = pd.DataFrame([new_match])

        # Encode categorical features
        for col in [
            "team1",
            "team2",
            "venue",
            "toss_winner",
            "toss_decision"
        ]:

            new_df[col] = le_dict[col].transform(
                new_df[col].astype(str)
            )

        # Convert to numpy
        new_data = new_df[
            [
                "team1",
                "team2",
                "venue",
                "toss_winner",
                "toss_decision",
                "team1_runs_last_5",
                "team2_runs_last_5"
            ]
        ].values

        # Scale only last two columns
        new_data[:, -2:] = scaler.transform(
            new_data[:, -2:]
        )

        # Prediction
        prediction_probability = model.predict(
            new_data,
            verbose=0
        )

        predicted_class = np.argmax(
            prediction_probability,
            axis=1
        )[0]

        predicted_team = le_winner.inverse_transform(
            [predicted_class]
        )[0]

        probability = prediction_probability[0][predicted_class] * 100

        st.success(
            f"🏆 Predicted Winner: {predicted_team}"
        )

        st.metric(
            "Winning Probability",
            f"{probability:.2f}%"
        )

        st.subheader("Prediction Probabilities")

        probability_df = pd.DataFrame({
            "Team": le_winner.classes_,
            "Probability": prediction_probability[0] * 100
        })

        probability_df = probability_df.sort_values(
            "Probability",
            ascending=False
        )

        st.bar_chart(
            probability_df.set_index("Team")
        )


# =========================================================
# ABOUT TEAMS PAGE
# =========================================================

elif page == "📈 About Teams":

    st.title("📈 IPL Team Statistics")

    if df.empty:

        st.warning(
            "No dataset found. Generate some data first."
        )

    else:

        selected_team = st.selectbox(
            "Select a team",
            TEAMS
        )

        # Matches involving selected team
        team_matches = df[
            (df["team1"] == selected_team) |
            (df["team2"] == selected_team)
        ]

        # Wins
        wins = len(
            df[df["winner"] == selected_team]
        )

        # Matches
        matches = len(team_matches)

        # Losses
        losses = matches - wins

        # Win percentage
        win_percentage = (
            wins / matches * 100
            if matches > 0
            else 0
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Matches", matches)

        with col2:
            st.metric("Wins", wins)

        with col3:
            st.metric("Losses", losses)

        with col4:
            st.metric(
                "Win %",
                f"{win_percentage:.2f}%"
            )

        st.divider()

        # Recent form
        st.subheader("📊 Recent Batting Form")

        team1_data = df[
            df["team1"] == selected_team
        ][["team1_runs_last_5"]].rename(
            columns={
                "team1_runs_last_5": "Runs"
            }
        )

        team2_data = df[
            df["team2"] == selected_team
        ][["team2_runs_last_5"]].rename(
            columns={
                "team2_runs_last_5": "Runs"
            }
        )

        runs_data = pd.concat(
            [team1_data, team2_data],
            ignore_index=True
        )

        if not runs_data.empty:

            st.line_chart(
                runs_data
            )

            st.metric(
                "Average Recent Runs",
                f"{runs_data['Runs'].mean():.1f}"
            )

        st.divider()

        # Win / Loss chart
        st.subheader("🏆 Wins vs Losses")

        result_data = pd.DataFrame({
            "Result": ["Wins", "Losses"],
            "Matches": [wins, losses]
        })

        st.bar_chart(
            result_data.set_index("Result")
        )


# =========================================================
# ABOUT PAGE
# =========================================================

elif page == "ℹ️ About":

    st.title("ℹ️ About This Project")

    st.write(
        """
        ### IPL Match Winner Prediction

        This project demonstrates how Machine Learning can be used
        to predict the possible winner of an IPL match.

        ### Machine Learning Model

        The prediction model is an Artificial Neural Network (ANN).

        ### Features Used

        - Team 1
        - Team 2
        - Venue
        - Toss Winner
        - Toss Decision
        - Team 1 recent batting form
        - Team 2 recent batting form

        ### Dataset

        The project uses synthetic IPL match data generated using Python.

        ### Technology

        - Python
        - Pandas
        - NumPy
        - Scikit-learn
        - TensorFlow / Keras
        - Streamlit
        """
    )

    st.info(
        "This project is created for learning Machine Learning, "
        "ANNs and Streamlit."
    )
    