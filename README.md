# ADY Streamlit Starter

Setup instructions (Windows PowerShell):

1. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If activation is blocked, you can run packages through the venv Python directly:

```powershell
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m streamlit run app.py
```

2. Install dependencies

```powershell
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
```

3. Run the Streamlit app

```powershell
.\.venv\Scripts\python -m streamlit run app.py
```

Notes:
- If you need database connectivity, install the appropriate driver (e.g. `pyodbc` for SQL Server, `mysql-connector-python` or `pymysql` for MySQL) and configure connection strings.
- To save your installed package versions: ` .\.venv\Scripts\python -m pip freeze > requirements.txt`
