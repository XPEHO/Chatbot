# Chatbot

## Export BookStack pages to PDF

### Prerequisites
- Python 3.9+
- BookStack API token (token id + token secret)
- BookStack URL

### Setup
1. Create a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Create a `.env` file (or update it) at the project root:
```env
BOOKSTACK_URL=your_bookstack_url
BOOKSTACK_TOKEN_ID=your_token_id
BOOKSTACK_TOKEN_SECRET=your_token_secret
```

## To run a notebook
```bash
pip install jupyter ipykernel
python -m ipykernel install --user --name chatbot-venv --display-name "Python (chatbot-venv)"
jupyter notebook
```

### Run the export
```bash
python export_pages.py
```

PDFs will be saved to `exports/`

# Pricing OPenAI
Pour un rechargement de base on utilise en moyenne 67 000 tokens soit 1/7 centimes.
Une question côut 1/40 centimes (une toute simple).
