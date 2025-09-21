import argparse, json
from pathlib import Path
import joblib, numpy as np, pandas as pd

def build_geojson(dfm, preds, risk):
    feats = [None] * len(dfm)
    for j, row in enumerate(dfm.itertuples(index=False)):
        feats[j] = {
            "type":"Feature",
            "geometry":{"type":"Point","coordinates":[float(row.lon_center), float(row.lat_center)]},
            "properties":{
                "cell_id": int(row.cell_id),
                "time": str(pd.to_datetime(row.time).date()),
                "risk": float(risk[j]),
                "prediction": float(preds[j]),
                "temp_c": float(row.mean_temp_c),
                "precip_mm": float(row.total_precip_mm),
                "humidity_pct": float(row.mean_rel_humidity_pct),
                "flood_recent": int(row.flood_recent),
            }
        }
    return {"type":"FeatureCollection","features":feats}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="dataset.csv")
    ap.add_argument("--model", default="models/xgb_dengue.pkl")
    ap.add_argument("--month", default="2024-07-01")
    ap.add_argument("--out", default="out_land/dengue/2024-07-01.geojson")   # write to out_land to avoid cache confusion
    ap.add_argument("--norm", action="store_true")
    args = ap.parse_args()

    bundle = joblib.load(args.model)
    model, feats, mu, sd = bundle["model"], bundle["feats"], bundle["mu"], bundle["sd"]

    df = pd.read_csv(args.csv, parse_dates=["time"])
    df["cell_id"] = df["cell_id"].astype("int32")
    dfm = df[df["time"] == pd.to_datetime(args.month)].copy()
    if dfm.empty:
        raise ValueError(f"No rows found for {args.month}")

    # ⛔ Require land_mask.csv (fail fast if missing)
    land = pd.read_csv("land_mask.csv")
    land["cell_id"] = land["cell_id"].astype("int32")
    dfm = dfm.merge(land, on="cell_id", how="left")
    before = len(dfm)
    dfm = dfm[dfm["is_land"] == 1].copy()
    dfm.drop(columns=["is_land"], inplace=True)
    print(f"[infer] Land filter: {before} -> {len(dfm)} rows")

    dfm.reset_index(drop=True, inplace=True)

    X = dfm[feats].replace([np.inf,-np.inf], np.nan).ffill().fillna(0.0)
    X = (X - mu[feats]) / sd[feats].replace(0,1.0)

    preds = model.predict(X).astype(float)
    risk = preds.copy()
    if args.norm:
        lo, hi = np.nanpercentile(preds, 2), np.nanpercentile(preds, 98)
        risk = np.clip((preds - lo) / (hi - lo + 1e-9), 0, 1)

    geo = build_geojson(dfm, preds, risk)
    outp = Path(args.out); outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(geo))
    print(f"✅ Wrote {outp}  features={len(geo['features'])}")

if __name__ == "__main__":
    main()
