from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


CHURN_THRESHOLD_DAYS = 180


FEATURE_COLUMNS = [
    #Time-based features
    "account_age_days",
    "days_since_last_activity",
    "days_since_last_repo_push",
    "days_since_last_event",

    #Ratio/proportion features
    "repos_per_year",
    "gists_per_year",
    "follower_ratio",
    "inactive_repo_ratio",
    "fork_repo_ratio",
    "archived_repo_ratio",
    "push_event_ratio",

    #Aggregation features
    "repo_count",
    "total_stars",
    "avg_stars_per_repo",
    "total_forks",
    "avg_forks_per_repo",
    "avg_repo_size",
    "avg_open_issues_per_repo",
    "language_count",
    "recent_event_count",
    "recent_push_event_count",
    "recent_commit_count",
    "avg_commits_per_week",
    "active_event_days",

    #Binary features
    "has_bio",
    "has_company",
    "has_location",
    "has_no_repos",
]


FEATURE_METADATA = [
    {
        "feature": "account_age_days",
        "type": "time-based",
        "reason": "Older accounts have had more time to create activity; age helps normalize behavior.",
    },
    {
        "feature": "days_since_last_activity",
        "type": "time-based",
        "reason": "Main recency signal; users inactive for a long time are more likely to churn.",
    },
    {
        "feature": "days_since_last_repo_push",
        "type": "time-based",
        "reason": "Measures how long it has been since the user's latest repository push.",
    },
    {
        "feature": "days_since_last_event",
        "type": "time-based",
        "reason": "Measures how long it has been since the user's latest public GitHub event.",
    },
    {
        "feature": "repos_per_year",
        "type": "ratio",
        "reason": "Normalizes repository count by account age.",
    },
    {
        "feature": "gists_per_year",
        "type": "ratio",
        "reason": "Normalizes public gist count by account age.",
    },
    {
        "feature": "follower_ratio",
        "type": "ratio",
        "reason": "Measures social engagement by comparing followers to following.",
    },
    {
        "feature": "inactive_repo_ratio",
        "type": "ratio",
        "reason": "Measures the proportion of repositories with no recent activity.",
    },
    {
        "feature": "fork_repo_ratio",
        "type": "ratio",
        "reason": "Measures how many repositories are forks instead of original projects.",
    },
    {
        "feature": "archived_repo_ratio",
        "type": "ratio",
        "reason": "A high archived ratio may indicate abandoned or inactive projects.",
    },
    {
        "feature": "push_event_ratio",
        "type": "ratio",
        "reason": "Measures how much of the user's recent activity is actual code pushing.",
    },
    {
        "feature": "repo_count",
        "type": "aggregation",
        "reason": "Counts the number of repositories collected for the user.",
    },
    {
        "feature": "total_stars",
        "type": "aggregation",
        "reason": "Total popularity signal across repositories.",
    },
    {
        "feature": "avg_stars_per_repo",
        "type": "aggregation",
        "reason": "Average popularity of the user's repositories.",
    },
    {
        "feature": "total_forks",
        "type": "aggregation",
        "reason": "Total reuse/collaboration signal across repositories.",
    },
    {
        "feature": "avg_forks_per_repo",
        "type": "aggregation",
        "reason": "Average collaboration or reuse per repository.",
    },
    {
        "feature": "avg_repo_size",
        "type": "aggregation",
        "reason": "Average size of the user's repositories.",
    },
    {
        "feature": "avg_open_issues_per_repo",
        "type": "aggregation",
        "reason": "Average number of open issues per repository.",
    },
    {
        "feature": "language_count",
        "type": "aggregation",
        "reason": "Counts how many different programming languages appear in the user's repositories.",
    },
    {
        "feature": "recent_event_count",
        "type": "aggregation",
        "reason": "Total number of recent public events.",
    },
    {
        "feature": "recent_push_event_count",
        "type": "aggregation",
        "reason": "Number of recent push events.",
    },
    {
        "feature": "recent_commit_count",
        "type": "aggregation",
        "reason": "Total commits found in recent public PushEvents.",
    },
    {
        "feature": "avg_commits_per_week",
        "type": "aggregation",
        "reason": "Estimated recent commit frequency per week.",
    },
    {
        "feature": "active_event_days",
        "type": "aggregation",
        "reason": "Counts how many different days the user had public events.",
    },
    {
        "feature": "has_bio",
        "type": "binary",
        "reason": "1 if the user has a profile bio, otherwise 0.",
    },
    {
        "feature": "has_company",
        "type": "binary",
        "reason": "1 if the user has company information, otherwise 0.",
    },
    {
        "feature": "has_location",
        "type": "binary",
        "reason": "1 if the user has location information, otherwise 0.",
    },
    {
        "feature": "has_no_repos",
        "type": "binary",
        "reason": "1 if the user has zero public repositories, otherwise 0.",
    },
]


def get_data_dirs() -> tuple[Path, Path]:
    """
    Inside Docker, use /data.
    Outside Docker, use local data/.
    """

    if Path("/data").exists():
        base_dir = Path("/data")
    else:
        base_dir = Path("data")

    raw_dir = base_dir / "raw"
    processed_dir = base_dir / "processed"

    processed_dir.mkdir(parents=True, exist_ok=True)

    return raw_dir, processed_dir


def read_csv_or_empty(path: Path, columns: list[str]) -> pd.DataFrame:
    """
    Read a CSV file safely.
    If it does not exist or is empty, return an empty DataFrame with expected columns.
    """

    if not path.exists():
        print(f"Warning: {path} was not found.")
        return pd.DataFrame(columns=columns)

    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        print(f"Warning: {path} is empty.")
        return pd.DataFrame(columns=columns)


def to_datetime_utc(series: pd.Series) -> pd.Series:
    """
    Convert a column to UTC datetime safely.
    Invalid values become NaT.
    """

    return pd.to_datetime(series, utc=True, errors="coerce")


def has_text(series: pd.Series) -> pd.Series:
    """
    Convert a text column into a binary 1/0 feature.
    1 = has text
    0 = empty or missing
    """

    return series.fillna("").astype(str).str.strip().ne("").astype(int)


def to_bool_int(series: pd.Series) -> pd.Series:
    """
    Convert boolean-like values into 1/0.
    Works even if CSV loaded True/False as text.
    """

    return (
        series.fillna(False)
        .astype(str)
        .str.lower()
        .isin(["true", "1", "yes"])
        .astype(int)
    )


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """
    Divide safely and replace infinity or missing values with 0.
    """

    result = numerator / denominator.replace(0, np.nan)
    result = result.replace([np.inf, -np.inf], np.nan)
    return result.fillna(0)


def normalize_user_columns(users_df: pd.DataFrame) -> pd.DataFrame:
    """
    Support both the improved scraper column names and the earlier minimal scraper names.
    """

    users_df = users_df.copy()

    if "user_created_at" not in users_df.columns and "created_at" in users_df.columns:
        users_df["user_created_at"] = users_df["created_at"]

    if "user_updated_at" not in users_df.columns and "updated_at" in users_df.columns:
        users_df["user_updated_at"] = users_df["updated_at"]

    if "days_since_last_activity" not in users_df.columns and "days_inactive" in users_df.columns:
        users_df["days_since_last_activity"] = users_df["days_inactive"]

    required_columns = [
        "username",
        "user_created_at",
        "user_updated_at",
        "public_repos",
        "public_gists",
        "followers",
        "following",
        "bio",
        "company",
        "location",
        "churned",
    ]

    for column in required_columns:
        if column not in users_df.columns:
            users_df[column] = np.nan

    return users_df


def build_user_features(users_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build features that only need user-level data.
    """

    users_df = normalize_user_columns(users_df)

    if users_df.empty:
        raise ValueError(
            "No user data found. Run scraper.py first to create github_users_labeled.csv."
        )

    if "churned" not in users_df.columns or users_df["churned"].isnull().all():
        raise ValueError(
            "No churned column found. Step 3 must create the churn label before Step 4."
        )

    now = datetime.now(timezone.utc)

    public_repos = pd.to_numeric(users_df["public_repos"], errors="coerce").fillna(0)
    public_gists = pd.to_numeric(users_df["public_gists"], errors="coerce").fillna(0)
    followers = pd.to_numeric(users_df["followers"], errors="coerce").fillna(0)
    following = pd.to_numeric(users_df["following"], errors="coerce").fillna(0)

    created_at = to_datetime_utc(users_df["user_created_at"])

    account_age_days = (now - created_at).dt.days
    account_age_days = account_age_days.clip(lower=1).fillna(1)

    account_age_years = account_age_days / 365.25

    if "last_activity_at" in users_df.columns:
        last_activity_at = to_datetime_utc(users_df["last_activity_at"])
    else:
        last_activity_at = to_datetime_utc(users_df["user_updated_at"])

    if "days_since_last_activity" in users_df.columns:
        days_since_last_activity = pd.to_numeric(
            users_df["days_since_last_activity"],
            errors="coerce",
        )

        calculated_days = (now - last_activity_at).dt.days
        days_since_last_activity = days_since_last_activity.fillna(calculated_days)
    else:
        days_since_last_activity = (now - last_activity_at).dt.days

    features_df = pd.DataFrame()

    features_df["username"] = users_df["username"]
    features_df["churned"] = pd.to_numeric(
        users_df["churned"],
        errors="coerce",
    ).fillna(0).astype(int)

    # Time-based
    features_df["account_age_days"] = account_age_days
    features_df["days_since_last_activity"] = days_since_last_activity

    # Ratio features
    features_df["repos_per_year"] = safe_divide(public_repos, account_age_years)
    features_df["gists_per_year"] = safe_divide(public_gists, account_age_years)
    features_df["follower_ratio"] = safe_divide(followers, following + 1)

    # Binary features
    features_df["has_bio"] = has_text(users_df["bio"])
    features_df["has_company"] = has_text(users_df["company"])
    features_df["has_location"] = has_text(users_df["location"])
    features_df["has_no_repos"] = (public_repos == 0).astype(int)

    return features_df


def build_repo_features(repos_df: pd.DataFrame, usernames: pd.Series) -> pd.DataFrame:
    """
    Build features from repository-level data.
    One output row per user.
    """

    output_df = pd.DataFrame({"username": usernames})

    repo_feature_columns = [
        "repo_count",
        "total_stars",
        "avg_stars_per_repo",
        "total_forks",
        "avg_forks_per_repo",
        "avg_repo_size",
        "avg_open_issues_per_repo",
        "language_count",
        "inactive_repo_ratio",
        "fork_repo_ratio",
        "archived_repo_ratio",
        "days_since_last_repo_push",
    ]

    if repos_df.empty:
        for column in repo_feature_columns:
            output_df[column] = 0
        return output_df

    repos_df = repos_df.copy()

    required_columns = [
        "username",
        "repo_name",
        "repo_pushed_at",
        "stargazers_count",
        "forks_count",
        "open_issues_count",
        "size",
        "language",
        "archived",
        "fork",
    ]

    for column in required_columns:
        if column not in repos_df.columns:
            repos_df[column] = np.nan

    now = datetime.now(timezone.utc)

    repos_df["repo_pushed_at"] = to_datetime_utc(repos_df["repo_pushed_at"])

    numeric_columns = [
        "stargazers_count",
        "forks_count",
        "open_issues_count",
        "size",
    ]

    for column in numeric_columns:
        repos_df[column] = pd.to_numeric(repos_df[column], errors="coerce").fillna(0)

    repos_df["archived_int"] = to_bool_int(repos_df["archived"])
    repos_df["fork_int"] = to_bool_int(repos_df["fork"])

    repos_df["days_since_repo_push"] = (
        now - repos_df["repo_pushed_at"]
    ).dt.days

    repos_df["repo_inactive"] = (
        repos_df["days_since_repo_push"].fillna(CHURN_THRESHOLD_DAYS + 1)
        > CHURN_THRESHOLD_DAYS
    ).astype(int)

    grouped = (
        repos_df.groupby("username")
        .agg(
            repo_count=("repo_name", "count"),
            total_stars=("stargazers_count", "sum"),
            avg_stars_per_repo=("stargazers_count", "mean"),
            total_forks=("forks_count", "sum"),
            avg_forks_per_repo=("forks_count", "mean"),
            avg_repo_size=("size", "mean"),
            avg_open_issues_per_repo=("open_issues_count", "mean"),
            language_count=("language", "nunique"),
            inactive_repo_ratio=("repo_inactive", "mean"),
            fork_repo_ratio=("fork_int", "mean"),
            archived_repo_ratio=("archived_int", "mean"),
            last_repo_push_at=("repo_pushed_at", "max"),
        )
        .reset_index()
    )

    grouped["days_since_last_repo_push"] = (
        now - grouped["last_repo_push_at"]
    ).dt.days

    grouped = grouped.drop(columns=["last_repo_push_at"])

    output_df = output_df.merge(grouped, on="username", how="left")

    for column in repo_feature_columns:
        if column not in output_df.columns:
            output_df[column] = 0

    output_df[repo_feature_columns] = output_df[repo_feature_columns].fillna(0)

    return output_df


def build_event_features(events_df: pd.DataFrame, usernames: pd.Series) -> pd.DataFrame:
    """
    Build features from recent public GitHub events.
    One output row per user.
    """

    output_df = pd.DataFrame({"username": usernames})

    event_feature_columns = [
        "recent_event_count",
        "recent_push_event_count",
        "push_event_ratio",
        "recent_commit_count",
        "avg_commits_per_week",
        "active_event_days",
        "days_since_last_event",
    ]

    if events_df.empty:
        for column in event_feature_columns:
            output_df[column] = 0
        return output_df

    events_df = events_df.copy()

    required_columns = [
        "username",
        "event_type",
        "event_created_at",
        "commit_count",
    ]

    for column in required_columns:
        if column not in events_df.columns:
            events_df[column] = np.nan

    now = datetime.now(timezone.utc)

    events_df["event_created_at"] = to_datetime_utc(events_df["event_created_at"])
    events_df["commit_count"] = pd.to_numeric(
        events_df["commit_count"],
        errors="coerce",
    ).fillna(0)

    events_df["is_push_event"] = (
        events_df["event_type"].fillna("").astype(str).eq("PushEvent")
    ).astype(int)

    events_df["event_day"] = events_df["event_created_at"].dt.date

    grouped = (
        events_df.groupby("username")
        .agg(
            recent_event_count=("event_type", "count"),
            recent_push_event_count=("is_push_event", "sum"),
            recent_commit_count=("commit_count", "sum"),
            active_event_days=("event_day", "nunique"),
            first_event_at=("event_created_at", "min"),
            last_event_at=("event_created_at", "max"),
        )
        .reset_index()
    )

    grouped["push_event_ratio"] = safe_divide(
        grouped["recent_push_event_count"],
        grouped["recent_event_count"],
    )

    event_window_days = (
        grouped["last_event_at"] - grouped["first_event_at"]
    ).dt.days + 1

    event_window_days = event_window_days.clip(lower=7).fillna(7)

    grouped["avg_commits_per_week"] = safe_divide(
        grouped["recent_commit_count"],
        event_window_days / 7,
    )

    grouped["days_since_last_event"] = (
        now - grouped["last_event_at"]
    ).dt.days

    grouped = grouped.drop(columns=["first_event_at", "last_event_at"])

    output_df = output_df.merge(grouped, on="username", how="left")

    for column in event_feature_columns:
        if column not in output_df.columns:
            output_df[column] = 0

    output_df[event_feature_columns] = output_df[event_feature_columns].fillna(0)

    return output_df

def clean_final_features(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace missing values and infinity values.
    Keep only username, feature columns, and churned label.
    """

    features_df = features_df.copy()

    for column in FEATURE_COLUMNS:
        if column not in features_df.columns:
            features_df[column] = 0

    features_df[FEATURE_COLUMNS] = features_df[FEATURE_COLUMNS].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    features_df[FEATURE_COLUMNS] = features_df[FEATURE_COLUMNS].fillna(0)

    for column in FEATURE_COLUMNS:
        features_df[column] = pd.to_numeric(
            features_df[column],
            errors="coerce",
        ).fillna(0)

    features_df["churned"] = pd.to_numeric(
        features_df["churned"],
        errors="coerce",
    ).fillna(0).astype(int)

    output_columns = ["username"] + FEATURE_COLUMNS + ["churned"]

    return features_df[output_columns]

def build_feature_dataset(
    users_df: pd.DataFrame,
    repos_df: pd.DataFrame,
    events_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the full user-level feature dataset.
    """

    user_features = build_user_features(users_df)
    repo_features = build_repo_features(repos_df, user_features["username"])
    event_features = build_event_features(events_df, user_features["username"])

    features_df = user_features.merge(repo_features, on="username", how="left")
    features_df = features_df.merge(event_features, on="username", how="left")

    features_df = clean_final_features(features_df)

    return features_df


def main() -> None:
    raw_dir, processed_dir = get_data_dirs()

    users_path = raw_dir / "github_users_labeled.csv"
    repos_path = raw_dir / "github_repos_raw.csv"
    events_path = raw_dir / "github_events_raw.csv"

    users_df = read_csv_or_empty(users_path, columns=[])
    repos_df = read_csv_or_empty(repos_path, columns=[])
    events_df = read_csv_or_empty(events_path, columns=[])

    features_df = build_feature_dataset(
        users_df=users_df,
        repos_df=repos_df,
        events_df=events_df,
    )

    features_output_path = processed_dir / "github_features.csv"
    metadata_output_path = processed_dir / "github_feature_metadata.csv"

    features_df.to_csv(features_output_path, index=False)
    pd.DataFrame(FEATURE_METADATA).to_csv(metadata_output_path, index=False)

    print("\nStep 4 completed successfully.")
    print(f"Feature dataset saved to: {features_output_path}")
    print(f"Feature metadata saved to: {metadata_output_path}")

    print("\nNumber of users:", len(features_df))
    print("Number of generated features:", len(FEATURE_COLUMNS))

    print("\nFeature columns:")
    for column in FEATURE_COLUMNS:
        print(f"- {column}")

    print("\nClass balance:")
    print(features_df["churned"].value_counts())

    print("\nPreview:")
    preview_columns = [
        "username",
        "days_since_last_activity",
        "repos_per_year",
        "follower_ratio",
        "inactive_repo_ratio",
        "recent_commit_count",
        "avg_commits_per_week",
        "has_no_repos",
        "churned",
    ]

    print(features_df[preview_columns].head())


if __name__ == "__main__":
    main()