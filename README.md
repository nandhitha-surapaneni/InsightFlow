# InsightFlow

AI-powered data analytics and business intelligence platform that automatically cleans datasets, performs statistical analysis, generates interactive visualizations, builds predictive models, and produces AI-generated business insights.

---

## Features

- Upload CSV datasets
- Automatic data cleaning
- Exploratory Data Analysis (EDA)
- Interactive charts and visualizations
- Machine Learning predictions
- AI-generated business insights
- Downloadable reports
- Modern React dashboard

---

## Tech Stack

### Frontend
- React
- TypeScript
- Vite
- Tailwind CSS

### Backend
- FastAPI
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Groq LLM

---

# Project Structure

```
InsightFlow/
│
├── backend/
│   ├── agents/
│   ├── main.py
│   ├── .env
│   ├── venv/ (or .venv/)
│   └── ...
│
├── frontend/
│
└── README.md
```

---

# Backend Setup

## 1. Navigate to backend

```bash
cd backend
```

## 2. Activate the virtual environment

If using `venv`

```bash
source venv/bin/activate
```

If using `.venv`

```bash
source .venv/bin/activate
```

## 3. Install dependencies

If a `requirements.txt` file exists:

```bash
pip install -r requirements.txt
```

Otherwise install the required packages manually:

```bash
pip install fastapi uvicorn pandas numpy scikit-learn matplotlib python-dotenv
```

Install any additional packages used by your project if needed.

## 4. Start the backend

```bash
uvicorn main:app --reload
```

Backend runs on

```
http://127.0.0.1:8000
```

---

# Frontend Setup

Navigate to the frontend folder

```bash
cd frontend
```

Install dependencies

```bash
npm install
```

Start the development server

```bash
npm run dev
```

Frontend runs on

```
http://localhost:3000
```

(or the port shown in the terminal)

---

# Workflow

1. Upload a CSV dataset
2. Backend cleans the data
3. AI agents analyze the dataset
4. Statistical insights are generated
5. Charts and visualizations are created
6. Machine learning models generate predictions
7. AI summarizes findings
8. Dashboard displays complete results

---

# Future Improvements

- Multiple dataset support
- More ML algorithms
- Export to PDF
- User authentication
- Cloud deployment
- Database integration

---
