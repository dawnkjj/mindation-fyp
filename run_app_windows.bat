@echo off
REM Run Mindation with the local virtual environment if it exists.
IF EXIST .venv\Scripts\python.exe (
  .venv\Scripts\python.exe -m streamlit run app.py
) ELSE (
  python -m streamlit run app.py
)
