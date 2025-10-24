# Disease-spread-prediction

> End-to-end ML pipeline for forecasting short-term disease case counts based on mobility, weather, and historical case data.  
> Built with *Python, **XGBoost, and **Flask* — ready for deployment as a prediction API.

---

## 🚀 Project Overview

This project predicts *next-day disease case counts* for different regions using synthetic or real-world data such as:

- *Mobility trends* (e.g., lockdown effects)  
- *Weather data* (temperature, humidity)  
- *Historical case counts*

The model learns temporal patterns (lags, rolling averages, and seasonal effects) to produce accurate short-term forecasts.

✅ *Goal:* Build a production-ready ML system from raw data to API inference.  
✅ *Use Case:* Public health forecasting, early warning systems, resource planning.  
✅ *Skills Demonstrated:* Data engineering, feature design, model tuning, interpretability, and ML deployment.

---

## 🧰 Tech Stack

| Category | Tools / Libraries |
|-----------|-------------------|
| *Languages* | Python 3.10+ |
| *Data Handling* | Pandas, NumPy |
| *Modeling* | XGBoost, Scikit-learn |
| *Visualization* | Matplotlib, SHAP |
| *Serving / API* | Flask |
| *Environment* | virtualenv / pip |

---

## 📂 Repository Structure

.
├── data_generation.py       # Generates synthetic dataset (mobility + weather + cases)
├── train_model.py           # Trains and evaluates XGBoost model
├── app.py                   # Flask REST API for predictions
├── requirements.txt          # Dependencies list
├── shap_summary.png          # Model interpretability plot (auto-generated)
└── README.md                 # You are here

---

## 🧪 Features & Workflow

1. *Data Generation / Collection*
   - Synthetic data simulates realistic patterns: seasonal weather, lockdowns, and infection trends.
   - Easily replaceable with real datasets (Google Mobility, WHO/CDC case reports, etc.).

2. *Feature Engineering*
   - Lag features (cases_lag1, cases_lag7, etc.)
   - Rolling averages (7-day, 14-day)
   - Date-time decomposition (day, month, weekday)
   - Region grouping and target shifting for forecasting

3. *Model Training*
   - Trained using *XGBoostRegressor* with hyperparameter tuning (GridSearchCV)
   - Group-based validation split to simulate unseen regions
   - Performance metrics: *MAE, **RMSE*
   - SHAP analysis for interpretability (shap_summary.png)

4. *Model Serving*
   - Flask API endpoint: /predict
   - Accepts JSON input with latest region data
   - Returns next-day predicted cases in real-time

---

## 🧾 Example Input / Output

*Request*
```bash
curl -X POST http://127.0.0.1:5000/predict \
-H "Content-Type: application/json" \
-d '{
  "region":"region_0",
  "date":"2025-01-05",
  "mobility":-12.3,
  "temperature":18.2,
  "humidity":60.5,
  "cases_lag1":45,
  "cases_lag2":40,
  "cases_lag3":38,
  "cases_lag7":30,
  "cases_lag14":20,
  "cases_roll_mean_7":41.0,
  "cases_roll_mean_14":36.5
}'

Response

{"prediction": 49.72}


⸻

⚙ How to Run Locally

1️⃣ Create environment and install dependencies

python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
pip install -r requirements.txt

2️⃣ Generate dataset

python data_generation.py --out data.csv --days 180

3️⃣ Train model

python train_model.py --data data.csv --model-path model.pkl

4️⃣ Run Flask API

python app.py --model-path model.pkl


⸻

📈 Model Evaluation

Metric	Description	Result (Example)
MAE	Mean Absolute Error	~4.2
RMSE	Root Mean Square Error	~6.1
Explainability	Top features: mobility, cases_lag1, humidity	via SHAP


⸻

💡 Key Highlights for Recruiters
	•	End-to-end pipeline: Data → Training → Evaluation → Deployment
	•	Temporal ML techniques (lags, rolling means, seasonal features)
	•	Explainable AI integration (SHAP visualization)
	•	Clean, production-grade structure (configurable scripts + REST API)
	•	Extensible design for real datasets (Google Mobility, WHO data, etc.)


Would you like me to make this *README automatically link to your Kaggle/GitHub profile* (with badges and stats panels like recruiters love)?  
If yes — drop your *GitHub username* and I’ll personalize the header section.
