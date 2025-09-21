# forecast_to_geojson.py
import argparse, json, os
from pathlib import Path
import joblib, numpy as np, pandas as pd

CLIMATE = ["mean_temp_c", "total_precip_mm", "mean_rel_humidity_pct", "flood_recent"]

def month_sin_cos(dt):
    m = dt.month
    return np.sin(2*np.pi*(m/12.0)), np.cos(2*np.pi*(m/12.0))

def next_month(date_str):
    dt = pd.to_datetime(date_str)
    dt = (dt + pd.offsets.MonthBegin(1)).normalize()
    return dt.strftime("%Y-%m-%d")

def build_climatology(df):
    # per cell_id & calendar month
    df["month"] = df["time"].dt.month
    aggs = {c:"mean" for c in CLIMATE}
    clim = df.groupby(["cell_id","month"], as_index=False).agg(aggs)
    # for flood_recent use mean as probability (0..1)
    return clim

def get_clim_row(clim, cell_id, month, col):
    row = clim[(clim.cell_id==cell_id) & (clim.month==month)]
    if row.empty: return np.nan
    return float(row.iloc[0][col])

def write_geojson(path, df_points, preds, risks, time_str):
    feats = []
    for i, r in df_points.iterrows():
        feats.append({
            "type":"Feature",
            "geometry":{"type":"Point","coordinates":[float(r.lon_center), float(r.lat_center)]},
            "properties":{
                "cell_id": int(r.cell_id),
                "time": time_str,
                "risk": float(risks[i]),
                "prediction": float(preds[i]),
                "temp_c": float(r["mean_temp_c"]),
                "precip_mm": float(r["total_precip_mm"]),
                "humidity_pct": float(r["mean_rel_humidity_pct"]),
                "flood_recent": int(round(float(r["flood_recent"]))),  # display-friendly
            }
        })
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({"type":"FeatureCollection","features":feats}))
    print(f"✅ Wrote {path}  features={len(feats)}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="dataset.csv")
    ap.add_argument("--disease", choices=["dengue","malaria","lyme"], default="dengue")
    ap.add_argument("--model", default="models/xgb_forecast_dengue.pkl")
    ap.add_argument("--start_month", default="2025-01-01")
    ap.add_argument("--horizon", type=int, default=6, help="number of future months to forecast")
    ap.add_argument("--outdir", default="out_forecast/dengue")
    ap.add_argument("--norm", action="store_true", help="normalize 2-98 pct to [0,1] for risk")
    args = ap.parse_args()

    # load resources
    bundle = joblib.load(args.model)
    model, feats, mu, sd = bundle["model"], bundle["feats"], bundle["mu"], bundle["sd"]

    df = pd.read_csv(args.csv, parse_dates=["time"])
    df["cell_id"] = df["cell_id"].astype("int32")

    # land-only
    land = pd.read_csv("land_mask.csv")
    land["cell_id"] = land["cell_id"].astype("int32")
    df = df.merge(land, on="cell_id", how="left")
    df = df[df["is_land"] == 1].drop(columns=["is_land"]).copy()

    # per-cell climatology
    clim = build_climatology(df)

    # last observed month & lag1 risk
    last_obs = df["time"].max().normalize()
    last_month_df = df[df["time"] == last_obs].copy()
    target_now = f"{args.disease}_risk_0_1"
    prev_risk = last_month_df[["cell_id", target_now]].set_index("cell_id")[target_now].astype(float)

    # static metadata per cell
    cells = last_month_df[["cell_id","lat_center","lon_center"]].drop_duplicates().set_index("cell_id")

    # step forward month by month
    month = pd.to_datetime(args.start_month).normalize()
    for step in range(args.horizon):
        mo = int(month.month)

        # assemble feature frame for all cells
        rows = []
        for cid, row in cells.iterrows():
            # base climate from climatology (current month)
            vals = {c: get_clim_row(clim, cid, mo, c) for c in CLIMATE}

            # lag1 climate: previous month (wrap around)
            mo_prev = 12 if mo==1 else mo-1
            for c in CLIMATE:
                vals[f"lag1_{c}"] = get_clim_row(clim, cid, mo_prev, c)

            # roll3 climate: average of month, prev, prev-1 (wrap)
            mo_prev2 = 11 if mo==1 else (12 if mo==2 else mo-2)
            for c in CLIMATE:
                v = np.nanmean([
                    get_clim_row(clim, cid, mo, c),
                    get_clim_row(clim, cid, mo_prev, c),
                    get_clim_row(clim, cid, mo_prev2, c)
                ])
                vals[f"roll3_{c}"] = v

            # lag1 risk: previous observed/predicted
            vals["lag1_risk"] = float(prev_risk.get(cid, np.nan))

            # month & coords
            s, c = month_sin_cos(month)
            vals["month_sin"], vals["month_cos"] = float(s), float(c)
            vals["lat_center"], vals["lon_center"] = float(row.lat_center), float(row.lon_center)
            vals["cell_id"] = int(cid)

            rows.append(vals)

        feat_df = pd.DataFrame(rows).sort_values("cell_id").reset_index(drop=True)

        # keep only features the model expects & scale with train stats
        X = feat_df[feats].replace([np.inf,-np.inf], np.nan).ffill().fillna(0.0)
        X = (X - mu[feats]) / sd[feats].replace(0,1.0)

        preds = model.predict(X).astype(float)
        risks = preds.copy()
        if args.norm:
            lo, hi = np.nanpercentile(preds, 2), np.nanpercentile(preds, 98)
            risks = np.clip((preds - lo) / (hi - lo + 1e-9), 0.0, 1.0)

        # write GeoJSON for this month
        # we also want to attach the climate numbers we used
        feat_df_display = feat_df.copy()
        feat_df_display["time"] = month.strftime("%Y-%m-%d")
        outpath = os.path.join(args.outdir, f"{month.strftime('%Y-%m-%d')}.geojson")
        write_geojson(outpath, feat_df_display, preds, risks, month.strftime("%Y-%m-%d"))

        # update prev_risk for next step (recursive)
        prev_risk = pd.Series(risks, index=feat_df["cell_id"].values)

        # advance month
        month = (month + pd.offsets.MonthBegin(1)).normalize()

if __name__ == "__main__":
    main()
