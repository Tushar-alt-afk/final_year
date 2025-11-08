import sys
import os
from pathlib import Path

# Add the backend directory to the system path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Set the TensorFlow logger to suppress warnings (optional but recommended)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Import and run the main application
from app import app

if __name__ == '__main__':
    # Flask will automatically use the settings configured in app.py
    app.run(debug=True, host='0.0.0.0', port=8000)