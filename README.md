# 🧠✍️ Neurodevelopmental Disorder Classification from Handwriting Analysis

A full-stack machine learning pipeline that analyzes handwriting samples to predict neurodevelopmental disorders using multiple state-of-the-art computer vision models.

## 🚀 Quick Start

```bash
# Clone and setup
git clone https://github.com/imnick48/NEURODEVELOPMENTAL_DISORDER_CLASSIFICATION.git
cd NEURODEVELOPMENTAL_DISORDER_CLASSIFICATION

# Create virtual environment
python3 -m venv .venv
source ./.venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Launch the app
python3 app.py
```

Open your browser to [http://127.0.0.1:5000](http://127.0.0.1:5000) to use the web interface.

## 🌟 Key Features

- **Multiple Model Architectures**:
  - Custom 3-layer CNN
  - LeNet-5
  - MobileNetV2
  - EfficientNet B0 & B3
  - Vision Transformer (ViT)

- **Interactive Web Interface**:
  - Upload handwriting samples
  - Select model architecture
  - View prediction results with confidence scores

## 🏗️ Project Structure

```
.
├── app.py                      # Flask application entry point
├── app/                        # Application modules
│   └── model/                  # Model implementations
├── static/                     # Static assets
│   ├── css/                    # Stylesheets
│   ├── js/                     # JavaScript files  
│   └── uploads/                # Uploaded handwriting samples
├── templates/                  # HTML templates
│   ├── base.html               # Base template
│   ├── index.html              # Main interface
│   └── result.html             # Results page
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

## 🛠️ Installation Details

### Prerequisites
- Python 3.9+
- pip package manager
- Virtual environment (recommended)

### Step-by-Step Setup
1. **Clone the repository**:
   ```bash
   git clone https://github.com/imnick48/NEURODEVELOPMENTAL_DISORDER_CLASSIFICATION.git
   cd NEURODEVELOPMENTAL_DISORDER_CLASSIFICATION
   ```

2. **Set up virtual environment**:
   ```bash
   python3 -m venv .venv
   source ./.venv/bin/activate  # On Windows use: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**:
   ```bash
   python3 app.py
   ```

## 📚 Documentation

For detailed documentation on model architectures, API endpoints, and development guidelines, see the [project wiki](https://github.com/imnick48/NEURODEVELOPMENTAL_DISORDER_CLASSIFICATION/wiki).
