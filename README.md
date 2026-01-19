# Obesity-prediction
Predicting Obesity levels of patient

Obesity, which causes physical and mental problems, is a global health problem with serious consequences. The prevalence of obesity is increasing steadily, and therefore, new research is needed that examines the influencing factors of obesity and how to predict the occurrence of the condition according to these factors.

# Obesity Prediction API

This project implements a **machine learning–based obesity prediction service** using **Logistic Regression** and exposes it via a **FastAPI** REST API.
The application supports **local execution using a virtual environment** and **containerized deployment using Docker**.

---

## Project Structure

```
Obesity-prediction/
│
├── app.py                         # FastAPI application
├── test.py                         # API test client
├── logreg_obesity_multiclass.joblib # Trained ML model
├── requirements.txt                # Python dependencies
├── Dockerfile                      # Container definition
└── README.md
```

---

## 1. Environment Setup (Virtual Environment)

### 1.1 Create a virtual environment

```bash
python -m venv venv
```

### 1.2 Activate the virtual environment

**Linux / macOS**

```bash
source venv/bin/activate
```

**Windows (PowerShell)**

```powershell
venv\Scripts\Activate.ps1
```

---

## 2. Install Dependencies

All required dependencies are listed in `requirements.txt`.

```bash
pip install -r requirements.txt
```

---

## 3. Run the Application Locally

Start the FastAPI server:

```bash
python main.py
```

or

```bash
uvicorn main:app --host 0.0.0.0 --port 9696
```

The API will be available at:

```
http://localhost:9696
```

Interactive API documentation:

* Swagger UI: `http://localhost:9696/docs`
* ReDoc: `http://localhost:9696/redoc`

---

## 4. Test the API

Use the provided test client:

```bash
python test.py
```

Or send a request manually:

```bash
curl -X POST http://localhost:9696/predict \
  -H "Content-Type: application/json" \
  -d '{"Gender":1,"Age":28,"Height":172,"Weight":75,"family_history_with_overweight":1,"FAVC":0,"FCVC":2,"NCP":3,"CAEC":1,"SMOKE":0,"CH2O":2,"SCC":1,"FAF":1,"TUE":2,"CALC":0,"MTRANS":3}'
```

---

## 5. Containerization with Docker

### 5.1 Build the Docker image

From the project root:

```bash
docker build -t obesity-api .
```

---

### 5.2 Run the container

```bash
docker run -p 9696:9696 obesity-api
```

The service will be accessible at:

```
http://localhost:9696
```

---

## 6. Stopping the Container

Press `CTRL + C` in the terminal
or run:

```bash
docker ps
docker stop <container_id>
```

---

## 7. Notes

* The model expects **feature names and order exactly as used during training**
* Inputs must be **numeric and pre-encoded**
* The API returns predicted class probabilities and class labels
* Docker image is based on `python:3.12-slim` for minimal footprint

---

## 8. Technologies Used

* Python 3.12
* FastAPI
* Scikit-learn
* Pandas
* Joblib
* Docker
* Uvicorn

