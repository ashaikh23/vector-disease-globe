import os, subprocess, pandas as pd

diseases = ["dengue","malaria","lyme"]
months = pd.date_range("2015-01-01","2024-12-01",freq="MS").strftime("%Y-%m-01").tolist()

for d in diseases:
    os.makedirs(f"out_land/{d}", exist_ok=True)
    model = f"models/xgb_{d}.pkl"
    for m in months:
        out = f"out_land/{d}/{m}.geojson"
        if os.path.exists(out): 
            print("skip", out); continue
        print("make", out)
        subprocess.check_call([
            "python","predict_to_geojson.py",
            "--model", model, "--month", m, "--out", out, "--norm"
        ])
print("✅ done")
