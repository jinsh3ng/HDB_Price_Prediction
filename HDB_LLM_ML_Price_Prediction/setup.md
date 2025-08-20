## Setup Instructions

1. In the HDB_LLM_ML_PRICE_PREDICTION directory, create a Python virtual environment and activate it:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate # The .venv activation command might differ depending on your operating system

2. Install the required packages

   ```bash
   pip install -r requirements.txt


3. Start the Streamlit Frontend
   
   ```bash
   cd Frontend
   python -m streamlit run main.py

4. start the FASTAPI Backend
   ```bash
   cd Backend
   python -m uvicorn main:app --reload
