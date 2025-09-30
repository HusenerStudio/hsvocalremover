@echo off
echo Starting Vocal Remover Web App...
echo.
echo Opening http://localhost:8501 in your browser...
echo Press Ctrl+C to stop the server
echo.

REM Start Streamlit with minimal output
python -m streamlit run web_app.py --server.port 8501 --server.address localhost --browser.gatherUsageStats false --logger.level error

echo.
echo Web app has stopped.
pause