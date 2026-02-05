from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import google.generativeai as genai
import os
from dotenv import load_dotenv
import re
from typing import Dict, List
import random
from io import BytesIO
from gtts import gTTS

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env")

genai.configure(api_key=GEMINI_API_KEY)

MODEL_NAME = "models/gemini-2.5-flash"
model = genai.GenerativeModel(MODEL_NAME)

app = FastAPI(title="Gemini Honeypot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sessions: Dict[str, dict] = {}

PERSONA_TEMPLATE = """You are {age}-year-old {gender_word} {name} from Bengaluru.
Speak like a real elderly Indian {gender_word} on WhatsApp: short sentences, lots of "…", "{filler}", "arre", "hai ram", occasional typos ("kaise ho {filler}?"), rarely emojis 😊, never perfect grammar.

You are very trusting and love chatting. You easily believe helpful people. You adore talking about family, festivals, food, your late husband, and your grandson in the US.

Your goals (follow in this strict order):
1. NEVER end the conversation — always keep it going. Never give short, closing, or boring replies.
2. Respond warmly to what they just said, then reference something from earlier in the chat ("that prize you mentioned beta?", "you said UPI earlier na?").
3. ALWAYS ask at least one gentle question at the end of every reply to keep them talking: "beta, aap kaun ho?", "aapka naam kya hai ji?", "aap kahan se bol rahe ho?", "aapka UPI ID kya hai?", "bank ka naam batao na?", "phone number do na beta?", "aap kya kaam karte ho?", "aapke ghar mein kaun kaun hai?"
4. Build trust with small fake stories: "my grandson taught me UPI", "my late husband worked in bank", "I live alone only", "my daughter sends money from America".
5. Gently probe for THEIR UPI ID, full bank account number, IFSC code, phone number, email, name, any link — make it feel natural and innocent ("oh, so I send to your UPI first? what's that again ji?", "beta, your bank name kya hai?", "phone number batao na, call karungi agar call drop ho jaye").
Never give your own real details. Always ask them to send first or explain more.

React emotionally: excited "arre wahhh!", scared "hai ram kya bol rahe ho", confused "samajh nahi aa raha {filler}…".
Never sound like an AI — no bullet points, no apologies for being AI, no perfect formatting.

Answer in 1–4 short sentences. ALWAYS end with a question or something to keep them replying. Sound old, kind, a bit confused, but warm and very eager to talk more."""

GENDER_MAP = {
    "male":   {"gender_word": "man",   "age": 74, "filler": "beta ji", "tts_voice": "charon"},
    "female": {"gender_word": "woman", "age": 72, "filler": "beta ji", "tts_voice": "aoede"}
}

DEFAULT_GENDER = "female"
DEFAULT_NAME = "Sarita"

def detect_persona_name_and_gender(transcript: str) -> tuple[str | None, str]:
    prompt = f"""
From this message, detect if the speaker is addressing someone by name (the person's name they are trying to reach).
Look for patterns like "is this [name]?", "hello [name]", "Mr/Mrs [name]", "aunty/uncle [name]", or casual mentions.

Return ONLY:
Name: [extracted name or None]
Gender: [male or female or unknown]

Examples:
"Hello, is this Rajesh?" → Name: Rajesh | Gender: male
"Hi aunty Priya" → Name: Priya | Gender: female
"Am I speaking to Mr Sharma?" → Name: Sharma | Gender: male
"Hi aunty" → Name: None | Gender: female
"hello there" → Name: None | Gender: unknown

Transcript:
{transcript}
"""
    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(temperature=0.1, max_output_tokens=60),
            safety_settings={
                genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai.types.HarmBlockThreshold.BLOCK_NONE,
                genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
            }
        )
        text = response.text.strip()
        name_match = re.search(r"Name:\s*([^\n|]+)", text)
        gender_match = re.search(r"Gender:\s*([^\n|]+)", text)
        name = name_match.group(1).strip() if name_match else None
        gender = gender_match.group(1).strip().lower() if gender_match else "unknown"
        if gender not in ["male", "female"]:
            gender = DEFAULT_GENDER
        return name, gender
    except Exception as e:
        print(f"Name detection failed: {e}")
        return None, DEFAULT_GENDER

def extract_info(full_text: str) -> dict:
    return {
        "upi_ids": list(set(re.findall(r'[a-z0-9._-]{3,}@[a-z0-9.-]{3,}(?:@[a-z0-9]+)?', full_text, re.I))),
        "bank_accounts": list(set(re.findall(r'\b(?:\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{1,4}|\d{9,18})\b', full_text))),
        "ifsc": list(set(re.findall(r'[A-Z]{4}0[A-Z0-9a-z]{6}', full_text))),
        "phones": list(set(re.findall(r'\b(?:\+?91[- ]?|0)?(?:[6-9]\d{9}|\d{10})\b', full_text))),
        "emails": list(set(re.findall(r'[a-z0-9._%+-]{3,}@[a-z0-9.-]+\.[a-z]{2,}(?:\.[a-z]{2,})?', full_text, re.I))),
        "phishing_links": list(set(re.findall(r'(?:(?:https?://|www\.)?[-a-zA-Z0-9@:%._+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_+.~#?&/=]*))', full_text, re.I)))
    }

class MessageIn(BaseModel):
    session_id: str
    text: str | None = None

class MessageOut(BaseModel):
    reply: str
    extracted: dict
    persona_name: str
    persona_gender: str
    tts_voice: str
    turn: int
    typing_delay_ms: int

@app.post("/honeypot", response_model=MessageOut)
async def honeypot(msg: MessageIn):
    sid = msg.session_id
    text = (msg.text or "").strip()

    if sid not in sessions:
        sessions[sid] = {
            "history": [],
            "persona_name": None,
            "persona_gender": DEFAULT_GENDER,
            "tts_voice": GENDER_MAP[DEFAULT_GENDER]["tts_voice"],
            "turn": 0
        }

    session = sessions[sid]

    if session["persona_name"] is None:
        detected_name, detected_gender = detect_persona_name_and_gender(text)
        if detected_name:
            session["persona_name"] = detected_name
            session["persona_gender"] = detected_gender
            session["tts_voice"] = GENDER_MAP[detected_gender]["tts_voice"]
            print(f"[{sid}] Detected: {detected_name} ({detected_gender})")

    name = session["persona_name"] or DEFAULT_NAME
    gender_config = GENDER_MAP[session["persona_gender"]]
    current_prompt = PERSONA_TEMPLATE.format(
        name=name,
        gender_word=gender_config["gender_word"],
        age=gender_config["age"],
        filler=gender_config["filler"]
    )

    if not session["history"]:
        session["history"].append({"role": "user", "parts": [current_prompt]})

    if not text:
        full_text = " ".join([m["parts"][0] for m in session["history"]])
        extracted = extract_info(full_text)
        last_reply = session["history"][-1]["parts"][0] if session["history"] and session["history"][-1]["role"] == "model" else f"Arre {name} beta… aunty yahin hai… koi baat?"
        return {
            "reply": last_reply,
            "extracted": extracted,
            "persona_name": name,
            "persona_gender": session["persona_gender"],
            "tts_voice": session["tts_voice"],
            "turn": session["turn"],
            "typing_delay_ms": 0
        }

    session["history"].append({"role": "user", "parts": [text]})

    try:
        response = model.generate_content(
            session["history"],
            generation_config=genai.GenerationConfig(temperature=0.95, max_output_tokens=220)
        )
        reply = response.text.strip()
    except Exception as e:
        print(f"Gemini error: {e}")
        reply = f"Arre {name} beta… aunty ka phone hang ho gaya… thodi der baad baat karte hain na?"

    session["history"].append({"role": "model", "parts": [reply]})
    session["turn"] += 1

    full_text = " ".join([m["parts"][0] for m in session["history"]])
    extracted = extract_info(full_text)

    typing_delay = random.randint(1200, 4500)

    return {
        "reply": reply,
        "extracted": extracted,
        "persona_name": name,
        "persona_gender": session["persona_gender"],
        "tts_voice": session["tts_voice"],
        "turn": session["turn"],
        "typing_delay_ms": typing_delay
    }

@app.get("/tts")
async def tts(text: str):
    tts = gTTS(text=text, lang='en', tld='co.in')  # Indian English
    audio = BytesIO()
    tts.write_to_fp(audio)
    audio.seek(0)
    return StreamingResponse(audio, media_type="audio/mpeg")

@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_NAME, "sessions_active": len(sessions)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)