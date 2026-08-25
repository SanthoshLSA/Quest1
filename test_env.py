"""
Environment & Dependency Sanity Check Script
Verifies that key packages (cv2, yt_dlp, rapidfuzz, PIL) can be imported successfully.
"""
import sys

def check_environment():
    print(f"Python Executable: {sys.executable}")
    print(f"Python Version   : {sys.version.split()[0]}")
    
    modules = {
        "cv2": "OpenCV",
        "yt_dlp": "yt-dlp",
        "rapidfuzz": "RapidFuzz",
        "PIL": "Pillow"
    }
    
    all_ok = True
    for module_name, display_name in modules.items():
        try:
            mod = __import__(module_name)
            version = getattr(mod, "__version__", "Available")
            print(f"  [OK] {display_name}: {version}")
        except ImportError as e:
            print(f"  [MISSING] {display_name}: Not installed ({e})")
            all_ok = False
            
    if all_ok:
        print("\nAll core environment checks PASSED!")
    else:
        print("\nSome dependencies are missing. Install via: pip install -r requirements.txt")

if __name__ == "__main__":
    check_environment()
