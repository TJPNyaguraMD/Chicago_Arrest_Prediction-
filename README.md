# Chicago_Arrest_Prediction-
###  Project Overview
This project builds a scalable end-to-end machine learning pipeline to predict arrest outcomes using the Chicago Crimes dataset from Google BigQuery. The dataset contains over 8 million crime incidents, making it both large-scale and highly imbalanced.
The objective is to design, train, and evaluate ML models that can:
•	Identify which crime incidents are most likely to result in arrest
•	Handle significant class imbalance (non-arrest vs arrest)
•	Scale preprocessing and model training to millions of rows
•	Deliver strong discriminative performance (ROC-AUC ≥ 0.91)
The final optimized model is an XGBoost GPU-accelerated classifier, achieving:
•	Accuracy: 0.89
•	ROC-AUC: 0.913
•	Arrest Precision: 0.84
•	Arrest Recall: 0.68
•	Arrest F1-score: 0.75
________________________________________
### Dataset Source: BigQuery
Data is retrieved directly from:
bigquery-public-data.chicago_crime.crime
Query fields:
•	Crime type, description, FBI code
•	Spatial features (beat, district)
•	Temporal features (year, month, hour, weekday)
•	Domestic indicator
•	Arrest indicator (target variable)
BigQuery is used for efficient ingestion of millions of rows.
________________________________________
### Project Architecture
chicago-crime-prediction/
│
├── README.md              ← Project documentation (this file)/n
├── report.pdf             ← Full project report/n
├── notebook.ipynb         ← End-to-end notebook (EDA + modeling)/n
├── data_prep.py           ← Preprocessing, encoding, feature engineering/n
└── model.py               ← XGBoost model training + evaluation
________________________________________
### Preprocessing Pipeline (data_prep.py)
Includes:
•	Light feature engineering (SQL extracted temporal attributes)
•	Target Encoding for high-cardinality variables
•	One-Hot Encoding for low-cardinality variables
•	Scaling numerical features
•	Train/test split with stratification
•	Returns processed matrices for ML models
The preprocessing is designed to scale efficiently to multi-million-row datasets.
________________________________________
### Models Implemented
1. Logistic Regression
•	Fast baseline
•	Limited non-linear performance
•	AUC ≈ 0.88
2. Random Forest
•	Non-linear ensemble model
•	Arrest precision: 0.91
•	Arrest recall: 0.62
•	AUC ≈ 0.912
3. XGBoost (GPU Accelerated) — Final Model
•	Strongest performer
•	AUC: 0.913
•	Balanced precision–recall after tuning
•	Scale_pos_weight used to address class imbalance
•	Trained using gpu_hist for high speed
________________________________________
### Final Model Performance
Class 0 (No Arrest)
  Precision: 0.90
  Recall:    0.96
  F1-score:  0.93

Class 1 (Arrest)
  Precision: 0.84
  Recall:    0.68
  F1-score:  0.75

Overall Accuracy: 0.89
ROC-AUC:          0.913
The model demonstrates strong discriminative power, handles imbalance effectively, and remains stable across metrics.
________________________________________
###  How to Run the Project
1. Clone the Repo
git clone https://github.com/yourusername/chicago-crime-prediction.git
cd chicago-crime-prediction
2. Install Dependencies
pip install -r requirements.txt
(If you want, I can generate a requirements.txt file too.)
3. Set Up Google BigQuery Credentials
Place your GCP service account JSON file somewhere safe and set:
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r"path_to_json.json"
4. Run the Notebook
Best for demonstration and reproducibility:
jupyter notebook
5. Or Run the Model Directly
python model.py
________________________________________
### Future Improvements
•	Add geospatial clustering (hotspots, density maps)
•	Include weather and socioeconomic contextual features
•	Use LightGBM for even faster GPU-accelerated modeling
•	Deploy model as an API using FastAPI or Flask
•	Introduce SHAP explainability to analyze feature impact
•	Build a dashboard (Plotly Dash / Streamlit) for interactive exploration
________________________________________
### Full Report
See report.pdf for:
•	Literature review
•	Full methodology
•	Detailed model comparison
•	Recommendations
•	Conclusion + limitations
________________________________________
### Acknowledgements
Data sourced via Google BigQuery Public Datasets.
Project built with Python, XGBoost, scikit-learn, and Pandas.
