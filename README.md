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

### Run the export
```bash
python export_pages.py
```

PDFs will be saved to `exports/`
