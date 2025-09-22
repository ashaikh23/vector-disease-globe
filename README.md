# 🌍 Vector Disease Globe

An interactive 3D visualization and forecasting tool for global vector-borne disease risk. Built with **CesiumJS** and **XGBoost**, this project allows users to explore, filter, and forecast disease risks (Dengue, Malaria, Lyme) across time and geography.

Developed as part of the **Health in Climate Hackathon 2025**.

---

## ✨ Features

* 🗺️ **3D Globe Visualization** – Disease risk data is plotted directly onto a CesiumJS-powered 3D Earth.
* 🎚️ **Time Slider** – Scroll through monthly data to visualize changing patterns.
* 📊 **Multi-Disease Support** – Switch between Dengue, Malaria, and Lyme risk predictions.
* 🔮 **Forecasting with XGBoost** – Predict future risk levels beyond observed datasets.
* 🎨 **Interactive Controls** –

  * Palette selector (different color schemes).
  * Top N% filter (see only the highest risk regions).
  * Adjustable dot size for clarity.
* 🌊 **Land Masking** – Risk points only appear on land, avoiding clutter in oceans.

---

## 🚀 Demo

* [🌐 Live Demo (Google Drive)](https://drive.google.com/file/d/134ldGZN7-PGTFalisL7epE_wPyR4tHgy/view?usp=sharing)
* [📑 Slide Deck](https://docs.google.com/presentation/d/1pult1PlBB8uQ-X0tNqOju_jqrn81LWrsNPmFLSlrfRY/edit?slide=id.g3490313be59_0_0#slide=id.g3490313be59_0_0)

---

## 🛠️ Tech Stack

**Languages & Frameworks**

* Python
* JavaScript, HTML, CSS

**Data & Visualization**

* CesiumJS
* GeoJSON
* Natural Earth Land Polygons

**ML / Forecasting**

* XGBoost
* Pandas, NumPy
* GeoPandas

---

## ⚙️ Setup & Installation

1. **Clone the repo**

   ```bash
   git clone https://github.com/yourusername/vector-disease-globe.git
   cd vector-disease-globe
   ```

2. **Create a virtual environment**

   ```bash
   conda create -n vectorglobe python=3.10
   conda activate vectorglobe
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---

## 🔧 Usage

### 1. Preprocess land polygons

Download [Natural Earth 50m land shapefiles](https://www.naturalearthdata.com/downloads/50m-physical-vectors/) and place them in `ne_50m_land/`. Then run:

```bash
python build_land_mask.py
```

### 2. Generate historical predictions

```bash
python train.py
python predict_to_geojson.py
```

### 3. Forecast future risk with XGBoost

```bash
python forecast_train.py
python forecast_to_geojson.py
```

### 4. Export all data at once

```bash
python export_all.py
```

### 5. Serve the globe locally

```bash
python -m http.server 8000
```

Then open [http://localhost:8000/index.html](http://localhost:8000/index.html).

---

## 📊 Example Workflow

1. Select **Dengue** from the dropdown.
2. Use the **time slider** to explore 2010–2025 data.
3. Apply the **Top N% filter** (e.g., Top 10%) to highlight highest-risk regions.
4. Switch to **Forecast mode** to see XGBoost-projected future risk.

---

## 🌟 Inspiration

Vector-borne diseases affect billions worldwide. This tool was built to help researchers, policymakers, and health professionals **understand and forecast global risk trends** through an interactive, predictive visualization.

---

## 🚧 Challenges

* Integrating **land masking** to keep oceans clean.
* Handling **large GeoJSON datasets** efficiently in CesiumJS.
* Building a **forecasting pipeline** that smoothly extends data into the future.

---

## 📈 Future Improvements

* Incorporate **real-time climate data** (temperature, rainfall, humidity).
* Add more diseases beyond Dengue, Malaria, and Lyme.
* Deploy on the **cloud** (Heroku / AWS) for public access.

---

## 👥 Team

* Abhinav Barat
* Aymaan Shaikh

---

## 📜 License

MIT License – feel free to fork, modify, and build on top of this project.
