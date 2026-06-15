# Customer Churn Predictor App

This project is a Dockerized machine learning web application that predicts whether a GitHub user is likely to churn. In this project, churn means that a user has been inactive for more than 180 days based on public GitHub activity.

The project uses GitHub API data, generates meaningful user activity features, compares four feature selection methods, trains a Random Forest model, and exposes the final model through a FastAPI `/predict` endpoint.

## Project Structure

```text
churn-predictor/
│
├── app/
│   ├── scraper.py
│   ├── features.py
│   ├── model.py
│   ├── main.py
│   └── model.pkl
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   └── eda_and_selection_verified_cesar.ipynb
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
└── report.pdf
```

## Main Files

| File                                               | Purpose                                                                    |
| -------------------------------------------------- | -------------------------------------------------------------------------- |
| `app/scraper.py`                                   | Fetches raw GitHub user, repository, and public event data.                |
| `app/features.py`                                  | Converts raw data into model-ready features.                               |
| `app/model.py`                                     | Trains, evaluates, saves, and loads the Random Forest model.               |
| `app/main.py`                                      | FastAPI application with `/health`, `/features`, and `/predict` endpoints. |
| `app/model.pkl`                                    | Saved trained model used by the API.                                       |
| `notebooks/eda_and_selection_verified_cesar.ipynb` | Notebook showing feature selection and analysis.                           |

## Churn Definition

A GitHub user is labeled as churned if their last public activity happened more than 180 days ago.

The scraper calculates the most recent activity using:

* user profile update date
* latest repository push date
* latest public GitHub event date

Then it creates:

```text
churned = 1 if days_since_last_activity > 180
churned = 0 otherwise
```

## Generated Features

The project generates several types of features:

### Time-based features

* `account_age_days`
* `days_since_last_activity`
* `days_since_last_repo_push`
* `days_since_last_event`

### Ratio features

* `repos_per_year`
* `gists_per_year`
* `follower_ratio`
* `inactive_repo_ratio`
* `fork_repo_ratio`
* `archived_repo_ratio`
* `push_event_ratio`

### Aggregation features

* `repo_count`
* `total_stars`
* `avg_stars_per_repo`
* `total_forks`
* `avg_forks_per_repo`
* `language_count`
* `recent_event_count`
* `recent_push_event_count`
* `recent_commit_count`
* `avg_commits_per_week`

### Binary features

* `has_bio`
* `has_company`
* `has_location`
* `has_no_repos`

## Feature Selection

The notebook compares four required feature selection methods:

1. Filter methods
2. Recursive Feature Elimination, or RFE
3. Decision Tree feature importance
4. Random Forest feature importance

The final selected features used by the API are:

```text
days_since_last_activity
account_age_days
language_count
has_bio
has_company
has_location
recent_event_count
```

## Model

The final model is a Random Forest classifier trained in `app/model.py`.

The model is saved as:

```text
app/model.pkl
```

The API loads this file when the application starts.

## Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY app/ .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## docker-compose.yml

```yaml
services:
  churn-api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/data
```

## How to Run the API

Make sure Docker Desktop is open.

From the project root folder, run:

```bash
docker-compose up --build
```

Then open the API documentation in your browser:

```text
http://localhost:8000/docs
```

## API Endpoints

### Health Check

```text
GET /health
```

Example response:

```json
{
  "status": "ok",
  "model_loaded": true
}
```

### Selected Features

```text
GET /features
```

Example response:

```json
{
  "selected_features": [
    "days_since_last_activity",
    "account_age_days",
    "language_count",
    "has_bio",
    "has_company",
    "has_location",
    "recent_event_count"
  ]
}
```

### Predict Churn

```text
POST /predict
```

Example input:

```json
{
  "days_since_last_activity": 300,
  "account_age_days": 2000,
  "language_count": 3,
  "has_bio": 1,
  "has_company": 0,
  "has_location": 1,
  "recent_event_count": 5
}
```

Example response:

```json
{
  "churned": false,
  "churn_probability": 0.35
}
```

The exact probability may change depending on the trained model.

## Curl Test

After running Docker, the endpoint can be tested with:

```bash
curl -X POST "http://localhost:8000/predict" \
-H "Content-Type: application/json" \
-d "{\"days_since_last_activity\":300,\"account_age_days\":2000,\"language_count\":3,\"has_bio\":1,\"has_company\":0,\"has_location\":1,\"recent_event_count\":5}"
```

## Example Inputs

### Likely inactive user

```json
{
  "days_since_last_activity": 600,
  "account_age_days": 2500,
  "language_count": 0,
  "has_bio": 0,
  "has_company": 0,
  "has_location": 0,
  "recent_event_count": 0
}
```

### More active user

```json
{
  "days_since_last_activity": 5,
  "account_age_days": 2000,
  "language_count": 5,
  "has_bio": 1,
  "has_company": 1,
  "has_location": 1,
  "recent_event_count": 40
}
```

## Optional: Re-run the Full Pipeline

The submitted API should run directly with Docker because `model.pkl` is already included in `app/`.

To rebuild the data and model from scratch, run:

```bash
docker-compose run --rm churn-api python scraper.py
docker-compose run --rm churn-api python features.py
```

Then train the model locally:

```bash
python app/model.py
```

Finally, rebuild the Docker image:

```bash
docker-compose up --build
```

## Notes and Limitations

This project uses public GitHub data only. Private repository activity is not visible, so some users may look inactive even if they are active privately.

The churn label is simulated using a 180-day inactivity threshold. This is useful for the project, but it is not the same as confirmed user cancellation.

The dataset used in this project is small, so the model should be understood as an academic prototype rather than a production-ready churn system.

## Author

Cesar Oyola

## Course

Introduction to Data Science
Informatics Engineering
