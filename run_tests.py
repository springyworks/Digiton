import subprocess
import sys
import os

def run_test(script_path):
    print(f"Running {script_path}...")
    result = subprocess.run([sys.executable, script_path], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"✓ {script_path} passed")
        return True
    else:
        print(f"✗ {script_path} failed")
        print(result.stderr)
        return False

def main():
    print("Running Regression Tests...")
    tests = [
        "tests/regression/test_modem.py",
        "tests/regression/test_chat.py",
        # Add other regression tests here as they are migrated
    ]
    
    passed = 0
    for test in tests:
        if run_test(test):
            passed += 1
            
    print(f"\nSummary: {passed}/{len(tests)} tests passed.")
    
    if passed == len(tests):
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
