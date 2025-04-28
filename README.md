# Flight Prices Analysis Project

This project downloads flight price data from Kaggle and stores it in a SQLite database for analysis.

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
pip install -r requirements-build.txt
```

2. Configure environment variables (optional):
Create a `.env` file or set environment variables directly:
```bash
KAGGLE_DATASET=dilwong/flightprices  # Optional: Override default dataset
DB_PATH=flights.db  # Optional: Override default database path
TOGETHER_API_KEY=your_together_api_key  # Your Together API key
```

3. Run the data loader:
```bash
python load_data.py
```

The script will download the flight prices dataset and create a SQLite database (default: `flights.db`).

4. Run the agent: (Accessible on [https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024](https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024))
```bash
langgraph dev
```


## Environment Variables

- `KAGGLE_DATASET`: Dataset to download (default: "dilwong/flightprices")
- `DB_PATH`: Path where the SQLite database will be created (default: "flights.db")
- `TOGETHER_API_KEY`: Your Together API key
