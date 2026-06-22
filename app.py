import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

st.set_page_config(
    page_title="FIFA World Cup Analytics Dashboard",
    page_icon="⚽",
    layout="wide"
)

st.title("🏆 FIFA World Cup Analytics Dashboard")
st.markdown(
    "Interactive Football Analytics & Match Prediction System"
)
@st.cache_data
def load_data():

    url = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"

    df = pd.read_csv(url)

    df["date"] = pd.to_datetime(df["date"])

    df["year"] = df["date"].dt.year

    return df

df = load_data()
df = df[df["tournament"] == "FIFA World Cup"]
st.sidebar.title("⚽ Navigation")

home_stats = df.groupby(
    "home_team"
).agg(
    avg_home_goals=(
        "home_score",
        "mean"
    )
).reset_index()

away_stats = df.groupby(
    "away_team"
).agg(
    avg_away_goals=(
        "away_score",
        "mean"
    )
).reset_index()

home_wins = df.groupby("home_team").apply(
    lambda x: (
        x["home_score"] > x["away_score"]
    ).mean()
).reset_index(name="home_win_rate")

away_wins = df.groupby("away_team").apply(
    lambda x: (
        x["away_score"] > x["home_score"]
    ).mean()
).reset_index(name="away_win_rate")

team_stats = (
    home_stats
    .merge(home_wins,on="home_team")
    .merge(
        away_stats,
        left_on="home_team",
        right_on="away_team",
        how="left"
    )
    .merge(
        away_wins,
        on="away_team",
        how="left"
    )
)

team_stats = team_stats.fillna(0)

def create_outcome(row):

    if row["home_score"] > row["away_score"]:
        return 2

    elif row["home_score"] == row["away_score"]:
        return 1

    return 0

df["outcome"] = df.apply(
    create_outcome,
    axis=1
)

model_df = df.copy()

model_df = model_df.merge(
    team_stats,
    left_on="home_team",
    right_on="home_team",
    how="left"
)



model_df = model_df.rename(
    columns={
        "avg_home_goals":"home_avg_goals",
        "home_win_rate":"home_win_rate"
    }
)

model_df = model_df.merge(
    team_stats[
        [
            "home_team",
            "avg_away_goals",
            "away_win_rate"
        ]
    ],
    left_on="away_team_x",
    right_on="home_team",
    how="left",
    suffixes=("","_y")
)







model_df = model_df.rename(
    columns={
        "avg_away_goals":"away_avg_goals",
        "away_win_rate":"away_win_rate"
    }
)




features = [
    "home_avg_goals",
    "home_win_rate",
    "away_avg_goals",
    "away_win_rate"
]

X = model_df[features]

y = model_df["outcome"]


X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=150,
    max_depth=10,
    random_state=42
)

model.fit(
    X_train,
    y_train
)

predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)



page = st.sidebar.radio(
    "Select Page",
    [
        "📊 Tournament Overview",
        "🌍 Team Analytics",
        "📈 Goal Trends",
        "🥊 Head-to-Head",
        "🤖 Model Performance",
        "🔮 Match Predictor",
        "🏅 Team Rankings",
        "📰 Recent Matches"
    ]
)


def create_features(df):

    df = df.copy()

    def outcome(row):
        if row["home_score"] > row["away_score"]:
            return 2
        elif row["home_score"] == row["away_score"]:
            return 1
        else:
            return 0

    df["outcome"] = df.apply(outcome, axis=1)

    return df

df_ml = create_features(df)

def get_team_stats(team):

    home = df[df["home_team"] == team]
    away = df[df["away_team"] == team]

    matches = len(home)+len(away)

    goals = (
        home["home_score"].sum()
        +
        away["away_score"].sum()
    )

    wins = (
        (home["home_score"] >
         home["away_score"]).sum()
        +
        (away["away_score"] >
         away["home_score"]).sum()
    )

    return matches, goals, wins

if page == "📊 Tournament Overview":

    st.header("Tournament Overview")

    total_matches = len(df)

    total_teams = len(
        pd.concat(
            [df["home_team"], df["away_team"]]
        ).unique()
    )

    avg_goals = (
        df["home_score"] + df["away_score"]
    ).mean()

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Matches",
        f"{total_matches:,}"
    )

    col2.metric(
        "Teams",
        total_teams
    )

    col3.metric(
        "Avg Goals",
        round(avg_goals,2)
    )

    st.subheader("Matches Per Year")

    yearly = (
        df.groupby("year")
        .size()
        .reset_index(name="matches")
    )

    fig = px.line(
        yearly,
        x="year",
        y="matches"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

elif page == "🌍 Team Analytics":

    st.header("Team Analytics")

    teams = sorted(
        pd.concat(
            [df["home_team"], df["away_team"]]
        ).unique()
    )

    selected_team = st.selectbox(
        "Select Team",
        teams
    )

    home_games = df[
        df["home_team"] == selected_team
    ]

    away_games = df[
        df["away_team"] == selected_team
    ]

    matches = len(home_games) + len(away_games)

    goals_scored = (
        home_games["home_score"].sum()
        +
        away_games["away_score"].sum()
    )

    wins = (
        (
            home_games["home_score"]
            >
            home_games["away_score"]
        ).sum()
        +
        (
            away_games["away_score"]
            >
            away_games["home_score"]
        ).sum()
    )

    win_rate = wins / matches * 100

    c1,c2,c3 = st.columns(3)

    c1.metric("Matches", matches)
    c2.metric("Goals", goals_scored)
    c3.metric("Win Rate", f"{win_rate:.1f}%")
    st.subheader("Compare Two Teams")

    col1,col2 = st.columns(2)

    team_a = col1.selectbox(
    "Team A",
    teams
    )

    team_b = col2.selectbox(
    "Team B",
    teams,
    index=1
    )
    a_matches,a_goals,a_wins = get_team_stats(team_a)
    b_matches,b_goals,b_wins = get_team_stats(team_b)

    comparison_df = pd.DataFrame({
        "Metric":[
            "Matches",
            "Goals",
            "Wins"
        ],
        team_a:[
            a_matches,
            a_goals,
            a_wins
        ],
        team_b:[
            b_matches,
            b_goals,
            b_wins
        ]
    })

    st.dataframe(
        comparison_df,
        use_container_width=True
    )

elif page == "📈 Goal Trends":

    st.header("📈 Goal Trends")

    df["total_goals"] = (
        df["home_score"]
        +
        df["away_score"]
    )

    goals_year = (
        df.groupby("year")
        ["total_goals"]
        .mean()
        .reset_index()
    )

    fig = px.line(
        goals_year,
        x="year",
        y="total_goals",
        markers=True
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    highest_year = goals_year.loc[
        goals_year["total_goals"].idxmax()
    ]

    st.success(
        f"Highest scoring World Cup: "
        f"{int(highest_year['year'])}"
        f" ({highest_year['total_goals']:.2f} goals)"
    )

elif page == "🥊 Head-to-Head":

    st.header("Head-to-Head Comparison")

    teams = sorted(
        pd.concat(
            [df["home_team"], df["away_team"]]
        ).unique()
    )

    col1,col2 = st.columns(2)

    team1 = col1.selectbox(
        "Team 1",
        teams
    )

    team2 = col2.selectbox(
        "Team 2",
        teams,
        index=1
    )

    h2h = df[
        (
            (df["home_team"]==team1)
            &
            (df["away_team"]==team2)
        )
        |
        (
            (df["home_team"]==team2)
            &
            (df["away_team"]==team1)
        )
    ]

    st.metric(
        "Matches Played",
        len(h2h)
    )

    st.dataframe(
        h2h[
            [
                "date",
                "home_team",
                "away_team",
                "home_score",
                "away_score"
            ]
        ]
    )

    home_stats = (
        df.groupby("home_team")
        .agg(
            avg_home_goals=("home_score","mean")
        )
        .reset_index()
    )

    away_stats = (
        df.groupby("away_team")
        .agg(
            avg_away_goals=("away_score","mean")
        )
        .reset_index()
    )
    team1_wins = (
    (
        (h2h["home_team"] == team1)
        &
        (h2h["home_score"] > h2h["away_score"])
    )
    |
    (
        (h2h["away_team"] == team1)
        &
        (h2h["away_score"] > h2h["home_score"])
    )
    ).sum()

    team2_wins = (
        (
            (h2h["home_team"] == team2)
            &
            (h2h["home_score"] > h2h["away_score"])
        )
        |
        (
            (h2h["away_team"] == team2)
            &
            (h2h["away_score"] > h2h["home_score"])
        )
    ).sum()

    draws = len(h2h) - team1_wins - team2_wins
    c1,c2,c3 = st.columns(3)

    c1.metric(team1, team1_wins)
    c2.metric("Draws", draws)
    c3.metric(team2, team2_wins)


elif page == "🤖 Model Performance":

    st.header("🤖 Model Performance")

    st.metric(
        "Accuracy",
        f"{accuracy*100:.2f}%"
    )

    st.metric(
        "Test Samples",
        len(X_test)
    )
    importance_df = pd.DataFrame({
    "Feature":features,
    "Importance":model.feature_importances_
    })

    importance_df = importance_df.sort_values(
        "Importance",
        ascending=False
    )

    fig = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Feature Importance"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )
    cm = confusion_matrix(
    y_test,
    predictions
    )

    fig = px.imshow(
        cm,
        text_auto=True,
        labels={
            "x":"Predicted",
            "y":"Actual"
        },
        x=[
            "Away Win",
            "Draw",
            "Home Win"
        ],
        y=[
            "Away Win",
            "Draw",
            "Home Win"
        ]
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )



elif page == "🔮 Match Predictor":

    st.header(
        "Predict Match Outcome"
    )

    teams = sorted(
    team_stats["home_team"].unique()
    )

    col1,col2 = st.columns(2)

    home_team = col1.selectbox(
        "Home Team",
        teams
    )

    away_team = col2.selectbox(
        "Away Team",
        teams,
        index=1
    )
    if home_team == away_team:

        st.warning(
        "Choose different teams."
        )
    else:
        home_data = team_stats[
        team_stats["home_team"]
        ==
        home_team
        ].iloc[0]

        away_data = team_stats[
        team_stats["home_team"]
        ==
        away_team
        ].iloc[0]

        prediction_input = pd.DataFrame([{
        "home_avg_goals":
            home_data["avg_home_goals"],

        "home_win_rate":
            home_data["home_win_rate"],

        "away_avg_goals":
            away_data["avg_away_goals"],

        "away_win_rate":
            away_data["away_win_rate"]
        }])

        probabilities = model.predict_proba(
        prediction_input
        )[0]
        results_df = pd.DataFrame({

        "Outcome":[
            "Away Win",
            "Draw",
            "Home Win"
        ],

        "Probability":probabilities
        })

        fig = px.bar(
        results_df,
        x="Outcome",
        y="Probability",
        color="Outcome"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )
        best = np.argmax(
        probabilities
        )

        outcomes = [
            "Away Win",
            "Draw",
            "Home Win"
        ]

        st.success(
        f"Predicted Result: "
        f"{outcomes[best]}"
        )
        st.subheader(
        "Why This Prediction?"
        )

        st.write(
            f"""
            • {home_team} average home goals:
            {home_data['avg_home_goals']:.2f}

            • {home_team} win rate:
            {home_data['home_win_rate']:.2%}

            • {away_team} average away goals:
            {away_data['avg_away_goals']:.2f}

            • {away_team} away win rate:
            {away_data['away_win_rate']:.2%}
            """
        )

elif page == "🏅 Team Rankings":


    st.header("🏅 Team Rankings")

    teams = pd.concat(
        [df["home_team"], df["away_team"]]
    ).unique()

    ranking_data = []

    for team in teams:

        home = df[df["home_team"] == team]
        away = df[df["away_team"] == team]

        played = len(home) + len(away)

        wins = (
            (home["home_score"] > home["away_score"]).sum()
            +
            (away["away_score"] > away["home_score"]).sum()
        )

        goals = (
            home["home_score"].sum()
            +
            away["away_score"].sum()
        )

        ranking_data.append(
            [team, played, wins, goals]
        )

    ranking_df = pd.DataFrame(
        ranking_data,
        columns=[
            "Team",
            "Matches",
            "Wins",
            "Goals"
        ]
    )

    ranking_df["Win Rate"] = (
        ranking_df["Wins"]
        /
        ranking_df["Matches"]
        * 100
    )

    ranking_df = ranking_df.sort_values(
        "Win Rate",
        ascending=False
    )

    st.dataframe(
        ranking_df,
        use_container_width=True
    )

    fig = px.bar(
        ranking_df.head(10),
        x="Team",
        y="Win Rate",
        title="Top Teams by Win Rate"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

elif page == "📰 Recent Matches":

    st.header("Recent Matches")

    recent = df.sort_values(
        "date",
        ascending=False
    ).head(20)

    st.dataframe(
        recent[
            [
                "date",
                "home_team",
                "away_team",
                "home_score",
                "away_score"
            ]
        ],
        use_container_width=True
    )

csv = df.to_csv(index=False)

st.download_button(
    label="📥 Download Dataset",
    data=csv,
    file_name="world_cup_data.csv",
    mime="text/csv"
)
st.markdown("---")

st.markdown(
"""
Developed by Sameer Danish

Built with:
- Streamlit
- Pandas
- Plotly
- Scikit-Learn
"""
)