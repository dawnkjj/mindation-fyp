# Environment checker for Mindation on Windows.
#check whether can run the app, and whether all dependencies are installed
from __future__ import annotations

# standard library imports
import importlib.util
import shutil
import sys

def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None

# Main function to check the setup
def main() -> None:
    print("Mindation setup check")
    print("=" * 30)
    print(f"Python: {sys.version.split()[0]}")

    # Check Python version
    if sys.version_info < (3, 10):
        print("[FAIL] Python 3.10+ is required. Install/use Python 3.12.")
    else:
        print("[OK] Python version is suitable.")

    # Check for required modules
    for module in ["streamlit", "numpy", "sklearn", "pypdf", "requests"]:
        print(f"[{'OK' if has_module(module) else 'MISSING'}] {module}")

    # Check for optional modules
    optional = {
        "whisper": "Whisper speech-to-text",
        "sentence_transformers": "SentenceTransformer semantic retrieval",
        "transformers": "BART/BLIP transformer models",
        "torch": "PyTorch backend",
        "PIL": "Pillow image support",
    }

    for module, label in optional.items():
        print(f"[{'OK' if has_module(module) else 'OPTIONAL MISSING'}] {label}")

    ffmpeg = shutil.which("ffmpeg")
    print(f"[{'OK' if ffmpeg else 'OPTIONAL'}] FFmpeg on PATH: {ffmpeg or 'not found; imageio-ffmpeg may still help'}")

    print("\nTo run: python -m streamlit run app.py")


if __name__ == "__main__":
    main()
