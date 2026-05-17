import os
import sys
import streamlit.web.cli as stcli

if __name__ == "__main__":
    # PyInstaller unpacks your files into a temporary folder called _MEIPASS.
    # We need to tell Streamlit to look for your app inside that temp folder.
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        bundle_dir = sys._MEIPASS
    else:
        bundle_dir = os.path.dirname(os.path.abspath(__file__))
        
    # The name of your actual Streamlit script
    script_path = os.path.join(bundle_dir, 'app_Doping.py')
    
    # Simulate the "streamlit run" command
    sys.argv = ["streamlit", "run", script_path, "--global.developmentMode=false"]
    sys.exit(stcli.main())
