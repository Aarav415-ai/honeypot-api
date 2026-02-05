# India Impact Buildathon - Problem 2: AI Honeypot for Scam Detection

A dynamic honeypot that pretends to be an elderly Indian aunty on WhatsApp to waste scammers' time, extract their UPI/bank details, and log intelligence in real-time.

Team: Sammit • Aarav • Maharth

## Features

- Uses Gemini 2.5 Flash to generate realistic, emotional replies
- Detects victim name/gender from scammer's first message
- Extracts UPI IDs, bank accounts, IFSC, phones, emails, phishing links
- Live dashboard showing extracted scam intelligence
- Conversation persists across messages (same session_id)
- Public API endpoint for external tools (e.g. voice with Deepgram)

## Tech Stack

- Backend: FastAPI + Google Gemini API
- Frontend: HTML + CSS + JavaScript (no framework)
- Deployment: Render.com (free tier)

## How to Run Locally

1. Clone the repo: 
    git clone https://github.com/yourusername/honeypot-ai.git
cd honeypot-ai
2. Create `.env` in `backend/` folder:
    GEMINI_API_KEY=your-gemini-api-key-here
text
3. Install dependencies (in backend folder):
    cd backend
    pip install -r requirements.txt
4. Start backend:
    py -m uvicorn main:app --reload --port 8000
5. Open frontend:
- Go to `frontend/index.html`
- Right-click → Open with Live Server (VS Code extension) or double-click

6. Test:
- Go to http://localhost:8000/docs → send POST to /honeypot
- Dashboard auto-refreshes every 6 seconds

## Example Scam Message (test in Swagger)

```json
{
"session_id": "test-001",
"text": "Hello aunty, this is Amit. Send UPI winner@paytm and bank 3456789012345678 IFSC SBIN0009876. Call +919876543210."
}