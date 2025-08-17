#!/usr/bin/env python3
"""
Wrapper to run Pinocchio scripts with clean exit to avoid memory cleanup issues.
"""

import sys
import os
import importlib.util

def run_script_with_clean_exit(script_path):
    """Run a Python script and exit cleanly to avoid memory cleanup issues."""
    try:
        # Load and execute the script
        spec = importlib.util.spec_from_file_location("__main__", script_path)
        if spec is None:
            print(f"Error: Could not load script {script_path}")
            os._exit(1)
        
        module = importlib.util.module_from_spec(spec)
        sys.modules["__main__"] = module
        
        # Execute the script
        spec.loader.exec_module(module)
        
        # Clean exit without Python cleanup
        os._exit(0)
        
    except SystemExit as e:
        # Handle normal sys.exit() calls
        os._exit(e.code if e.code is not None else 0)
    except Exception as e:
        print(f"Error running script: {e}")
        import traceback
        traceback.print_exc()
        os._exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 clean_runner.py <script.py>")
        os._exit(1)
    
    script_path = sys.argv[1]
    if not os.path.isfile(script_path):
        print(f"Error: Script '{script_path}' not found")
        os._exit(1)
    
    run_script_with_clean_exit(script_path)
