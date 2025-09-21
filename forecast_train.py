# forecast_train.py
import argparse, os, joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score

CLIMATE = ["mean_temp_c", "total_precip_mm", "mean_rel_humidity_pct", "flood_recent"]

def add_time_features(df):
    # month cyclic features
    m = df["time"].dt.month
    df["month_sin"] = np.sin(2*np.pi*(m/12.0))
    df["month_cos"] = np.cos(2*np.pi*(m/12.0))
    return df

def add_lags_and_target(df, disease):
    # sort per cell
    df = df.sort_values(["cell_id","time"]).copy()
    g = df.groupby("cell_id", group_keys=False)

    # lag-1 of climate and rolling means (3 months)
    for c in CLIMATE:
        df[f"lag1_{c}"] = g[c].shift(1)
        df[f"roll3_{c}"] = g[c].rolling(3, min_periods=2).mean().reset_index(level=0, drop=True)

    # previous month risk (strong signal)
    target_now = f"{disease}_risk_0_1"
    df["lag1_risk"] = g[target_now].shift(1)

    # target is next-month risk
    df["y_next"] = g[target_now].shift(-1)

    # drop rows without needed lags/target
    need = ["lag1_risk"] + [f"lag1_{c}" for c in CLIMATE]
    df = df.dropna(subset=need + ["y_next"]).copy()
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="dataset.csv")
    ap.add_argument("--disease", choices=["dengue","malaria","lyme"], default="dengue")
    ap.add_argument("--train_until", default="2022-12-01")
    ap.add_argument("--val_until",   default="2023-12-01")
    ap.add_argument("--out", default="models/xgb_forecast_dengue.pkl")
    args = ap.parse_args()

    # load + land-only
    df = pd.read_csv(args.csv, parse_dates=["time"])
    df["cell_id"] = df["cell_id"].astype("int32")
    try:
        land = pd.read_csv("land_mask.csv")
        land["cell_id"] = land["cell_id"].astype("int32")
        before = len(df)
        df = df.merge(land, on="cell_id", how="left")
        df = df[df["is_land"] == 1].drop(columns=["is_land"]).copy()
        print(f"[train-forecast] land filter: {before:,} -> {len(df):,}")
    except FileNotFoundError:
        print("[train-forecast] land_mask.csv not found — using all cells.")

    df = add_time_features(df)
    df = add_lags_and_target(df, args.disease)

    # feature list
    base_feats = CLIMATE + [f"lag1_{c}" for c in CLIMATE] + [f"roll3_{c}" for c in CLIMATE]
    extra = ["lag1_risk","lat_center","lon_center","month_sin","month_cos"]
    feats = [c for c in base_feats + extra if c in df.columns]

    # time split
    df = df.sort_values(["time","cell_id"])
    tr = df[df["time"] <= args.train_until]
    va = df[(df["time"] > args.train_until) & (df["time"] <= args.val_until)]
    te = df[df["time"] > args.val_until]

    # scale with train stats
    def clean(x): return x.replace([np.inf,-np.inf], np.nan).ffill().fillna(0.0)
    tr[feats], va[feats], te[feats] = clean(tr[feats]), clean(va[feats]), clean(te[feats])
    mu, sd = tr[feats].mean(), tr[feats].std().replace(0,1.0)
    z = lambda d: (d[feats]-mu)/sd

    Xtr, ytr = z(tr).values, tr["y_next"].values
    Xva, yva = z(va).values, va["y_next"].values
    Xte, yte = z(te).values, te["y_next"].values

    model = XGBRegressor(
        n_estimators=800, learning_rate=0.05, max_depth=8,
        subsample=0.9, colsample_bytree=0.9, objective="reg:squarederror", n_jobs=-1
    )
    model.fit(Xtr, ytr, eval_set=[(Xva,yva)], verbose=False)

    def score(name,X,y):
        p = model.predict(X); print(f"{name}: MAE={mean_absolute_error(y,p):.4f}  R2={r2_score(y,p):.3f}")
    score("Train", Xtr, ytr); score("Valid", Xva, yva); score("Test ", Xte, yte)

    os.makedirs("models", exist_ok=True)
    bundle = {
        "model": model,
        "feats": feats,
        "mu": mu,
        "sd": sd,
        "disease": args.disease,
        "last_history_month": str(df["time"].max().date())
    }
    joblib.dump(bundle, args.out)
    print("✅ Saved", args.out)

if __name__ == "__main__":
    main()
