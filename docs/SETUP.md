# ⚙️ PharmaLens Setup Guide

This guide describes how to configure your local environment, download the public FDA datasets, and run the PharmaLens platform.

---

## 🛠️ Step 1: Environment Setup

The project requires **Python 3.9** or higher. It is recommended to run the dashboard inside a isolated virtual environment.

### 1. Create a Virtual Environment
In your terminal, navigate to the repository root directory and run:

```bash
# macOS/Linux
python3 -m venv .venv

# Windows
python -m venv .venv
```

### 2. Activate the Environment
```bash
# macOS/Linux
source .venv/bin/activate

# Windows (Command Prompt)
.venv\Scripts\activate.bat

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

---

## 📦 Step 2: Install Dependencies

With the virtual environment active, run the following command to install the required packages:

```bash
pip install -r requirements.txt
```

---

## 🗄️ Step 3: Download FAERS Datasets

Because the raw ASCII event files are too large for standard Git tracking (exceeding 100 MB limits), they are excluded via `.gitignore`. You must obtain them manually:

1. Visit the [FDA FAERS Data Page](https://open.fda.gov/data/faers/).
2. Locate the **2025 Q4** ASCII files zip package.
3. Download and extract the package.
4. Copy the following flat files and paste them directly into the **root directory** of this repository:
   * `DEMO25Q4.txt` (Demographics table)
   * `DRUG25Q4.txt` (Medication history)
   * `OUTC25Q4.txt` (Patient outcome flags)
   * `REAC25Q4.txt` (Symptoms and reactions)

*Note: Ensure files are named exactly as shown above (case-sensitive) and are placed in the root directory alongside `app.py`.*

---

## 🖥️ Step 4: Run the Application

Start the Streamlit local dashboard by executing:

```bash
streamlit run app.py
```

Streamlit will automatically open the dashboard in your default web browser at:
`http://localhost:8501`
