#fetch raw data from chosen API

import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests


BASE_URL = "https://api.github.com"

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

HEADERS = {
    "Accept": "application/vnd.github+json",
    "User-Agent": "churn-predictor-project",
}

if GITHUB_TOKEN:
    HEADERS["Authorization"] = f"Bearer {GITHUB_TOKEN}"


USERNAMES = [ #20 users x 3 request/user = 60 requests
    "torvalds",
    "mojombo",
    "pjhyett",
    "wycats",
    "defunkt",
    "kennethreitz",
    "octocat",
    "gaearon",
    "addyosmani",
    "sindresorhus",
    "tj",
    "JakeWharton",
    "yyx990803",
    "antirez",
    "martinfowler",
    "mdo",
    "dhh",
    "sebmarkbage",
    "getify",
    "matz",
]


REQUEST_DELAY = 1.0
CHURN_THRESHOLD_DAYS = 180
MAX_REPO_PAGES_PER_USER = 1


def get_data_dir() -> Path:
    """
    Inside Docker, save to /data/raw.
    Outside Docker, save to data/raw.
    """

    if Path("/data").exists():
        data_dir = Path("/data/raw")
    else:
        data_dir = Path("data/raw")

    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def github_get(url: str, params: dict | None = None):
    """
    Make a GET request to GitHub and return JSON data.
    """

    try:
        response = requests.get(
            url,
            headers=HEADERS,
            params=params,
            timeout=15,
        )
    except requests.RequestException as error:
        print(f"Request failed: {error}")
        return None

    if response.status_code == 403:
        print("GitHub rate limit or forbidden request.")
        print("Try again later or use a GitHub token.")
        return None

    if response.status_code != 200:
        print(f"Request failed with status {response.status_code}: {url}")
        return None

    return response.json()


def fetch_user_profile(username: str) -> dict | None:
    """
    Fetch basic public data for one GitHub user.
    """

    url = f"{BASE_URL}/users/{username}"
    data = github_get(url)

    if data is None:
        return None

    return {
        "username": username,
        "user_type": data.get("type"),
        "user_created_at": data.get("created_at"),
        "user_updated_at": data.get("updated_at"),
        "public_repos": data.get("public_repos", 0),
        "public_gists": data.get("public_gists", 0),
        "followers": data.get("followers", 0),
        "following": data.get("following", 0),
        "bio": data.get("bio"),
        "company": data.get("company"),
        "location": data.get("location"),
    }


def fetch_user_repos(username: str) -> list[dict]:
    """
    Fetch repository data for one GitHub user.
    """

    all_repos = []

    for page in range(1, MAX_REPO_PAGES_PER_USER + 1):
        url = f"{BASE_URL}/users/{username}/repos"

        params = {
            "per_page": 100,
            "page": page,
            "sort": "pushed",
            "direction": "desc",
        }

        data = github_get(url, params=params)

        if not data:
            break

        for repo in data:
            all_repos.append(
                {
                    "username": username,
                    "repo_name": repo.get("name"),
                    "repo_full_name": repo.get("full_name"),
                    "repo_created_at": repo.get("created_at"),
                    "repo_updated_at": repo.get("updated_at"),
                    "repo_pushed_at": repo.get("pushed_at"),
                    "stargazers_count": repo.get("stargazers_count", 0),
                    "forks_count": repo.get("forks_count", 0),
                    "watchers_count": repo.get("watchers_count", 0),
                    "open_issues_count": repo.get("open_issues_count", 0),
                    "size": repo.get("size", 0),
                    "language": repo.get("language"),
                    "archived": repo.get("archived", False),
                    "fork": repo.get("fork", False),
                    "has_issues": repo.get("has_issues", False),
                }
            )

        time.sleep(REQUEST_DELAY)

    return all_repos


def fetch_user_events(username: str) -> list[dict]:
    """
    Fetch recent public events for one GitHub user.
    PushEvent records can contain commits.
    """

    url = f"{BASE_URL}/users/{username}/events/public"

    params = {
        "per_page": 100,
    }

    data = github_get(url, params=params)

    if not data:
        return []

    events = []

    for event in data:
        payload = event.get("payload", {})
        commits = payload.get("commits", [])

        events.append(
            {
                "username": username,
                "event_type": event.get("type"),
                "event_created_at": event.get("created_at"),
                "repo_name": event.get("repo", {}).get("name"),
                "commit_count": len(commits) if isinstance(commits, list) else 0,
            }
        )

    return events


def create_churn_labels(
    users_df: pd.DataFrame,
    repos_df: pd.DataFrame,
    events_df: pd.DataFrame,
    threshold_days: int = CHURN_THRESHOLD_DAYS,
) -> pd.DataFrame:
    """
    Create user-level churn labels.

    churned = 1 means inactive for more than threshold_days.
    churned = 0 means active within threshold_days.
    """

    labeled_df = users_df.copy()

    labeled_df["last_activity_at"] = pd.to_datetime(
        labeled_df["user_updated_at"],
        utc=True,
        errors="coerce",
    )

    if not repos_df.empty:
        repos_df = repos_df.copy()

        repos_df["repo_pushed_at"] = pd.to_datetime(
            repos_df["repo_pushed_at"],
            utc=True,
            errors="coerce",
        )

        repo_last_activity = (
            repos_df.groupby("username")["repo_pushed_at"]
            .max()
            .reset_index()
            .rename(columns={"repo_pushed_at": "last_repo_push_at"})
        )

        labeled_df = labeled_df.merge(
            repo_last_activity,
            on="username",
            how="left",
        )

        labeled_df["last_activity_at"] = labeled_df[
            ["last_activity_at", "last_repo_push_at"]
        ].max(axis=1)

    if not events_df.empty:
        events_df = events_df.copy()

        events_df["event_created_at"] = pd.to_datetime(
            events_df["event_created_at"],
            utc=True,
            errors="coerce",
        )

        event_last_activity = (
            events_df.groupby("username")["event_created_at"]
            .max()
            .reset_index()
            .rename(columns={"event_created_at": "last_event_at"})
        )

        labeled_df = labeled_df.merge(
            event_last_activity,
            on="username",
            how="left",
        )

        labeled_df["last_activity_at"] = labeled_df[
            ["last_activity_at", "last_event_at"]
        ].max(axis=1)

    now = datetime.now(timezone.utc)

    labeled_df["days_since_last_activity"] = (
        now - labeled_df["last_activity_at"]
    ).dt.days

    labeled_df["churned"] = (
        labeled_df["days_since_last_activity"] > threshold_days
    ).astype(int)

    return labeled_df


def main() -> None:
    users_records = []
    repos_records = []
    events_records = []

    for username in USERNAMES:
        print(f"\nFetching data for: {username}")

        profile = fetch_user_profile(username)

        if profile is None:
            print(f"Skipping {username}")
            continue

        users_records.append(profile)

        repos = fetch_user_repos(username)
        repos_records.extend(repos)

        events = fetch_user_events(username)
        events_records.extend(events)

        time.sleep(REQUEST_DELAY)

    users_df = pd.DataFrame(users_records)
    repos_df = pd.DataFrame(repos_records)
    events_df = pd.DataFrame(events_records)

    labeled_users_df = create_churn_labels(
        users_df=users_df,
        repos_df=repos_df,
        events_df=events_df,
        threshold_days=CHURN_THRESHOLD_DAYS,
    )

    data_dir = get_data_dir()

    users_df.to_csv(data_dir / "github_users_raw.csv", index=False)
    repos_df.to_csv(data_dir / "github_repos_raw.csv", index=False)
    events_df.to_csv(data_dir / "github_events_raw.csv", index=False)
    labeled_users_df.to_csv(data_dir / "github_users_labeled.csv", index=False)

    print("\nSaved files:")
    print(data_dir / "github_users_raw.csv")
    print(data_dir / "github_repos_raw.csv")
    print(data_dir / "github_events_raw.csv")
    print(data_dir / "github_users_labeled.csv")

    print("\nUser records:", len(users_df))
    print("Repository records:", len(repos_df))
    print("Event records:", len(events_df))

    print("\nMissing values in users:")
    print(users_df.isnull().sum())

    print("\nClass balance:")
    print(labeled_users_df["churned"].value_counts())

    print("\nPreview:")
    print(
        labeled_users_df[
            [
                "username",
                "last_activity_at",
                "days_since_last_activity",
                "churned",
            ]
        ].head()
    )


if __name__ == "__main__":
    main()