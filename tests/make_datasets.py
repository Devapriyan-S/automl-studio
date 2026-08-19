"""Generate deliberately awkward datasets to prove the engine is dataset-agnostic."""
import numpy as np, pandas as pd

def binary_churn(n=600, seed=0):
    """Mixed types + missing values + an ID column + a date column."""
    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 72, n)
    monthly = rng.normal(70, 25, n).clip(15, 200)
    plan = rng.choice(["Basic", "Pro", "Enterprise"], n, p=[.5, .35, .15])
    city = rng.choice([f"City_{i}" for i in range(40)], n)
    signup = pd.to_datetime("2020-01-01") + pd.to_timedelta(rng.integers(0, 1500, n), "D")
    logit = -2.5 + 0.05*monthly - 0.06*tenure + (plan == "Basic")*1.2
    churn = rng.binomial(1, 1/(1+np.exp(-logit)))
    df = pd.DataFrame({
        "customer_id": [f"CUST-{i:05d}" for i in range(n)],   # identifier
        "tenure_months": tenure, "monthly_charge": monthly.round(2),
        "plan": plan, "city": city, "signup_date": signup.strftime("%Y-%m-%d"),
        "region": "APAC",                                      # constant
        "churned": np.where(churn == 1, "Yes", "No"),
    })
    df.loc[rng.choice(n, n//8, replace=False), "monthly_charge"] = np.nan
    df.loc[rng.choice(n, n//12, replace=False), "plan"] = np.nan
    return df, "churned"

def multiclass_iris_like(n=450, seed=1):
    rng = np.random.default_rng(seed)
    cls = rng.integers(0, 3, n)
    df = pd.DataFrame({
        "f1": rng.normal(cls*2.0, 0.8), "f2": rng.normal(cls*-1.5, 0.9),
        "f3": rng.normal(5, 2, n), "grade": [f"G{c}" for c in cls],
    })
    return df, "grade"

def regression_housing(n=700, seed=2):
    rng = np.random.default_rng(seed)
    area = rng.normal(1400, 450, n).clip(300, 4000)
    rooms = rng.integers(1, 7, n)
    loc = rng.choice(["North", "South", "East", "West"], n)
    price = 50_000 + 180*area + 25_000*rooms + pd.Series(loc).map(
        {"North": 40_000, "South": 0, "East": 15_000, "West": 60_000}).values \
        + rng.normal(0, 30_000, n)
    return pd.DataFrame({"area_sqft": area.round(), "rooms": rooms,
                         "locality": loc, "price": price.round()}), "price"

def text_sentiment(n=400, seed=3):
    rng = np.random.default_rng(seed)
    pos = ["excellent product highly recommend", "loved it works perfectly",
           "great quality fast delivery service", "amazing value for money"]
    neg = ["terrible waste of money", "broke after one day awful",
           "very poor quality do not buy", "disappointed refund requested please"]
    lab = rng.integers(0, 2, n)
    return pd.DataFrame({
        "review": [rng.choice(pos if l else neg) + f" item{rng.integers(99)}" for l in lab],
        "verified": rng.choice([True, False], n),
        "sentiment": np.where(lab == 1, "positive", "negative"),
    }), "sentiment"

def imbalanced_fraud(n=800, seed=4):
    rng = np.random.default_rng(seed)
    fraud = rng.binomial(1, 0.06, n)          # ~6% positives
    return pd.DataFrame({
        "amount": np.abs(rng.normal(100 + fraud*400, 90, n)).round(2),
        "n_attempts": rng.poisson(1 + fraud*3, n),
        "channel": rng.choice(["web", "app", "pos"], n),
        "is_fraud": fraud,
    }), "is_fraud"

def tiny_wide(n=40, seed=5):
    """Few rows, many columns — the classic CV-blowup case."""
    rng = np.random.default_rng(seed)
    d = {f"x{i}": rng.normal(0, 1, n) for i in range(25)}
    d["label"] = (d["x0"] + d["x1"] > 0).astype(int)
    return pd.DataFrame(d), "label"

ALL = {
    "binary_churn": binary_churn, "multiclass": multiclass_iris_like,
    "regression": regression_housing, "text": text_sentiment,
    "imbalanced": imbalanced_fraud, "tiny_wide": tiny_wide,
}
