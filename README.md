# Customer Segmentation API with FastAPI & Machine Learning

This project implements a **Customer Segmentation API** using **FastAPI** and a **KMeans clustering model** built with **scikit-learn**. The API allows you to send customer data and get the predicted cluster for segmentation.

---

## 🚀 Features

- Train a **KMeans clustering model** for customer segmentation.
- Preprocess features using **StandardScaler**.
- Expose a **REST API** with FastAPI to predict cluster from input data.
- Include **Pydantic models** for input validation.
- Serve predictions via a **POST `/predict-cluster`** endpoint.
- Optionally visualize clusters using **PCA** (in notebook).

---

## 📁 Project Structure

CustomerSegmentation/
├── main.py # FastAPI application
├── model.pkl # Trained KMeans model
├── scaler.pkl # StandardScaler used for preprocessing
├── CustomerSegmentation.ipynb # Jupyter notebook for training & exploration
├── requirements.txt # Python dependencies
└── README.md # Project documentation


---

## 🛠️ Installation

1. Clone the repository:

```bash
git clone https://github.com/<username>/customer-segmentation-fastapi.git
cd customer-segmentation-fastapi
```
# Create a virtual environment:

python -m venv env


# Activate the environment:

Windows:

.\env\Scripts\activate


Mac/Linux:

source env/bin/activate


# Install dependencies:

pip install -r requirements.txt

⚡ Running the API

Start the FastAPI server:

uvicorn main:app --reload


The API will be available at:

http://127.0.0.1:8000


Swagger UI: http://127.0.0.1:8000/docs

Redoc: http://127.0.0.1:8000/redoc

# 🔹 API Usage
POST /predict-cluster

Send customer data in JSON format:

{
  "Age": 65,
  "Income": 88000,
  "Total_Spending": 200,
  "NumWebPurchases": 5,
  "NumStorePurchases": 10,
  "NumWebVisitsMonth": 7,
  "Recency": 30
}


Response:

{
  "cluster": 5
}

# 📊 Model Training (Notebook)

The Jupyter notebook (CustomerSegmentation.ipynb) contains:

Data loading and preprocessing

Feature engineering (Total_Spending, Total_Children, AgeGroup)

KMeans clustering model training

PCA visualization for clusters

Save trained objects with pickle:

pickle.dump(kmeans, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

# ⚙️ Dependencies

Python 3.10+

FastAPI

Uvicorn

scikit-learn

pandas

numpy

pydantic

matplotlib & seaborn (for notebook visualization)

Install all dependencies:

pip install -r requirements.txt

# 🔒 Notes

Make sure model.pkl and scaler.pkl are in the same folder as main.py.

Input values should be within the same range as training data, otherwise KMeans may assign all inputs to the same cluster.



# 📌 Author

Mahmudul Hasan


⭐ License

This project is open-source and free to use.


---

