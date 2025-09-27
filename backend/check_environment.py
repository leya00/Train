import sys
import pkg_resources
import importlib

def check_package(package_name):
    try:
        importlib.import_module(package_name)
        return True
    except ImportError:
        return False

def main():
    required_packages = [
        'fastapi',
        'uvicorn', 
        'opencv-python',
        'ultralytics',
        'python-multipart',
        'numpy'
    ]
    
    print("Python version:", sys.version)
    print("\nChecking required packages:")
    
    all_installed = True
    for package in required_packages:
        is_installed = check_package(package.replace('-', '_'))
        status = "✓" if is_installed else "✗"
        print(f"{status} {package}")
        if not is_installed:
            all_installed = False
    
    if not all_installed:
        print("\nSome packages are missing. Please install them using:")
        print("pip install -r requirements.txt")
        sys.exit(1)
    else:
        print("\nAll required packages are installed!")

if __name__ == "__main__":
    main() 