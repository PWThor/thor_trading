#!/usr/bin/env python
import os
import subprocess
import sys
import platform

def main():
    """Launch the Streamlit dashboard."""
    # Get the path to the dashboard script
    dashboard_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'dashboard.py'
    )
    
    # Construct command to run Streamlit
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        dashboard_path,
        "--server.port=8501",
        "--browser.serverAddress=localhost"
    ]
    
    print("\n==========================================")
    print("THOR TRADING BACKTEST DASHBOARD")
    print("==========================================\n")
    print("Starting dashboard server...")
    print("Access the dashboard at: http://localhost:8501\n")
    
    # Run the Streamlit server
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nDashboard server stopped.")
    except Exception as e:
        print(f"\nError starting dashboard: {str(e)}")
        
        # Check if Streamlit is installed
        try:
            import streamlit
            print(f"Streamlit version: {streamlit.__version__}")
        except ImportError:
            print("\nStreamlit not found! Please install it with:")
            print("pip install streamlit plotly")
            
            # Offer to install
            if input("\nInstall now? (y/n): ").lower() == 'y':
                subprocess.run([sys.executable, "-m", "pip", "install", "streamlit", "plotly"])
                print("\nPackages installed. Please run this script again.")

if __name__ == "__main__":
    main()