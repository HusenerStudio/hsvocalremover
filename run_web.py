#!/usr/bin/env python3
"""
Web Application Launcher for Vocal Remover

This script launches the Streamlit web application with proper configuration
and handles startup automatically.

Usage:
    python run_web.py
"""

import subprocess
import sys
import os
import time
import webbrowser
from pathlib import Path


def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import streamlit
        import torch
        import torchaudio
        print("✅ All core dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install dependencies with: pip install -r requirements.txt")
        return False


def check_models():
    """Check if any trained models are available"""
    models_dir = Path("models")
    if models_dir.exists():
        model_files = list(models_dir.glob("*.pth")) + list(models_dir.glob("*.onnx"))
        if model_files:
            print(f"✅ Found {len(model_files)} model(s)")
            for model in model_files:
                print(f"   - {model.name}")
            return True
    
    print("⚠️  No trained models found in 'models/' directory")
    print("   You can:")
    print("   1. Upload models through the web interface")
    print("   2. Train a model: python main.py train --data_dir your_data/")
    return False


def is_port_in_use(port):
    """Check if a port is already in use"""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0


def start_streamlit(port=8501, quiet=True):
    """Start Streamlit server"""
    
    # Check if port is already in use
    if is_port_in_use(port):
        print(f"⚠️  Port {port} is already in use")
        print(f"   The web app might already be running at http://localhost:{port}")
        return None
    
    print(f"🚀 Starting web application on port {port}...")
    
    # Prepare command
    cmd = [
        sys.executable, "-m", "streamlit", "run", "web_app.py",
        "--server.port", str(port),
        "--server.address", "localhost",
        "--browser.gatherUsageStats", "false",
        "--server.headless", "true"
    ]
    
    if quiet:
        # Suppress most output
        cmd.extend([
            "--logger.level", "error",
            "--client.showErrorDetails", "false"
        ])
    
    try:
        # Start the process
        if quiet:
            # Redirect output to suppress messages
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            )
        else:
            process = subprocess.Popen(cmd)
        
        # Wait a moment for startup
        time.sleep(3)
        
        # Check if process is still running
        if process.poll() is None:
            print(f"✅ Web application started successfully!")
            print(f"   URL: http://localhost:{port}")
            return process
        else:
            stderr_output = process.stderr.read().decode() if process.stderr else "Unknown error"
            print(f"❌ Failed to start web application: {stderr_output}")
            return None
            
    except Exception as e:
        print(f"❌ Error starting web application: {e}")
        return None


def open_browser(url, delay=2):
    """Open browser after a delay"""
    time.sleep(delay)
    try:
        webbrowser.open(url)
        print(f"🌐 Opened browser to {url}")
    except Exception as e:
        print(f"⚠️  Could not open browser automatically: {e}")
        print(f"   Please open {url} manually")


def main():
    """Launch the web application"""
    print("🎵 Vocal Remover Web Application Launcher")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    # Check models (warning only)
    check_models()
    
    # Start Streamlit
    port = 8501
    process = start_streamlit(port, quiet=True)
    
    if process is None:
        return 1
    
    # Open browser
    url = f"http://localhost:{port}"
    open_browser(url)
    
    print("\n" + "=" * 50)
    print("🎉 Vocal Remover is now running!")
    print(f"   Web Interface: {url}")
    print("   Press Ctrl+C to stop the server")
    print("=" * 50)
    
    try:
        # Keep the script running and monitor the process
        while True:
            if process.poll() is not None:
                print("\n⚠️  Web application has stopped")
                break
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n👋 Stopping web application...")
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        print("✅ Web application stopped")
        return 0
    
    return 0


if __name__ == "__main__":
    exit(main())