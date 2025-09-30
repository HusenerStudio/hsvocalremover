#!/usr/bin/env python3
"""
Simple Vocal Remover Launcher
Just starts Streamlit - no complications!
"""

import subprocess
import sys

def main():
    print("🎵 Starting Vocal Remover...")
    print("   Web interface will open at: http://localhost:8501")
    print("   Press Ctrl+C to stop")
    print()
    
    try:
        # Simple Streamlit command
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "web_app.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 Stopped!")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have streamlit installed: pip install streamlit")

if __name__ == "__main__":
    main()