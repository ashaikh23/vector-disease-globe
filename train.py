import argparse, joblib, pandas as pd, numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score

NON_FEATURES = {
    "cell_id","lat_center","lon_center","time",
    "dengue_risk_0_1","malaria_risk_0_1","lyme_risk_0_1",
    "dengue_cases_per_100k_synth","malaria_cases_per_100k_synth","lyme_cases_per_100k_synth"
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="dataset.csv")
    ap.add_argument("--disease", choices=["dengue","malaria","lyme"], default="dengue")
    ap.add_argument("--target", choices=["cases","risk"], default="cases")
    ap.add_argument("--train_until", default="2021-12-01")
    ap.add_argument("--val_until",   default="2023-12-01")
    ap.add_argument("--out", default="models/xgb_model.pkl")
    args = ap.parse_args()

    df = pd.read_csv(args.csv, parse_dates=["time"]).sort_values(["time","cell_id"])
    target = f"{args.disease}_{'cases_per_100k_synth' if args.target=='cases' else 'risk_0_1'}"

    tr = df[df["time"] <= args.train_until].copy()
    va = df[(df["time"] > args.train_until) & (df["time"] <= args.val_until)].copy()
    te = df[df["time"] > args.val_until].copy()

    drop = set(NON_FEATURES) | {target}
    feats = [c for c in df.columns if c not in drop and np.issubdtype(df[c].dtype, np.number)]

    def clean(x): return x.replace([np.inf,-np.inf], np.nan).fillna(method="ffill").fillna(0.0)
    tr[feats], va[feats], te[feats] = clean(tr[feats]), clean(va[feats]), clean(te[feats])

    mu, sd = tr[feats].mean(), tr[feats].std().replace(0,1.0)
    z = lambda d: (d[feats]-mu)/sd
    Xtr, ytr = z(tr).values, tr[target].values
    Xva, yva = z(va).values, va[target].values
    Xte, yte = z(te).values, te[target].values

    model = XGBRegressor(
        n_estimators=900, learning_rate=0.05, max_depth=8,
        subsample=0.8, colsample_bytree=0.8, objective="reg:squarederror", n_jobs=-1
    )
    model.fit(Xtr, ytr, eval_set=[(Xva,yva)], verbose=False)

    def score(name,X,y):
        p = model.predict(X)
        print(f"{name}: MAE={mean_absolute_error(y,p):.3f} R2={r2_score(y,p):.3f}")
    score("Train",Xtr,ytr); score("Valid",Xva,yva); score("Test ",Xte,yte)

    import os; os.makedirs("models", exist_ok=True)
    joblib.dump({"model":model,"feats":feats,"mu":mu,"sd":sd,"target":target}, args.out)
    print("Saved model to", args.out)

if __name__ == "__main__":
    main()
