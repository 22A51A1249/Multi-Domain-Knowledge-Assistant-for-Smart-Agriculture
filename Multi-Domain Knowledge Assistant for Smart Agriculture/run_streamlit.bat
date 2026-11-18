@echo off
title AgriMind AI - Streamlit Launcher
echo ========================================================
echo   🌾 Launching AgriMind AI Streamlit Application...
echo ========================================================

if exist "venv\Scripts\streamlit.exe" (
    echo Starting Streamlit via virtual environment...
    venv\Scripts\streamlit.exe run streamlit_app.py
) else (
    echo Starting Streamlit via system python...
    streamlit run streamlit_app.py
)

pause
