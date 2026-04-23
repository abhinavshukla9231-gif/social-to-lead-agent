# AutoStream Social-to-Lead Agent

A terminal-based conversational AI agent that turns social media interest into qualified business leads using LangGraph and Google Gemini. Built for the Inflx platform by ServiceHive, this agent handles real conversations — understanding intent, answering product questions from a local knowledge base, and collecting lead details before triggering a CRM action.

---

## Features

* Graph-based conversation flow using LangGraph for clean, modular routing between nodes.
* Gemini-powered intent detection and RAG responses grounded in a local knowledge base.
* Sequential lead qualification that collects name, email, and platform one step at a time.
* Mock CRM tool that fires only after all three lead fields are confirmed — no premature triggers.
* Persistent state across conversation turns so context is never lost mid-flow.

---

## Tech Stack

* LLM Framework: LangGraph
* Model Backend: Google Gemini 2.5 Flash-Lite
* Language: Python 3.10+

---

## Setup & Installation

**1. Clone the repository**

```bash
git clone https://github.com/your-username/social-to-lead-agent.git
cd social-to-lead-agent
```

**2. Create a virtual environment**

```bash
python -m venv .venv

# Windows:
.venv\Scripts\activate

# Mac/Linux:
source .venv/bin/activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Configure environment variables**

* Copy `.env.example` to `.env`
* Add your Gemini API key

```
GEMINI_API_KEY=your_key_here
```

Get a free key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey) — no credit card needed.

**5. Run the agent**

```bash
python main.py
```

---

## Project Structure

```
Social-to-Lead Agentic Bot/
├── main.py            # Agent logic, graph nodes, routers, chat loop
├── knowledge.md       # Local knowledge base used for RAG
├── requirements.txt   # Project dependencies
├── .env.example       # Environment variable template
└── README.md          # This file
```

---

## Example Conversation

```
==================================================
AutoStream Assistant (LangGraph + Gemini)
Type 'exit' to quit.
==================================================

You: hi
Bot: Welcome to AutoStream — your AI-powered video editing platform for content creators.

I can help you with:
- Pricing and Plans
- Features and Policies
- Getting started

What would you like to know?

You: what are the pricing plans?
Bot: AutoStream has two plans:

Basic Plan
- $29/month
- 10 videos/month
- 720p resolution

Pro Plan
- $79/month
- Unlimited videos
- 4K resolution
- AI captions
- 24/7 support

You: I want to subscribe to the Pro plan for my YouTube channel
Bot: I would like to get you set up. Could you share your name?

You: Abhinav Sharma
Bot: Thank you, Abhinav Sharma. What is your email address?

You: abhinav@gmail.com
Bot: Which platform do you create content on? (e.g., YouTube, Instagram, TikTok)

You: YouTube

Lead captured successfully: Abhinav Sharma, abhinav@gmail.com, Youtube

Bot: You are all set, Abhinav Sharma. We have captured your details and our team
will reach out to your Youtube account shortly. Welcome to AutoStream.
```

---

## Architecture Explanation

The agent runs on **LangGraph**, which made sense here because the conversation has a very specific structure — it needs to move between greeting, answering questions, collecting data, and firing a tool, without any of those steps bleeding into each other. A simple chain would not handle that cleanly. LangGraph lets you define each step as a node and route between them with full control.

**Why LangGraph over AutoGen?**
AutoGen works well for multi-agent setups where agents collaborate, but this project is a single-agent flow with strict routing rules. LangGraph's conditional edges made it straightforward to say "only start collecting lead info if the user explicitly shows high intent" and "never fire the CRM tool until all three fields are confirmed."

**State Management**
There is one `AgentState` dictionary that travels through every node and stays alive across the entire conversation in the `run_chat()` loop. It holds the user's name, email, platform, intent, response, and a `collecting_lead` flag that keeps the lead flow on track even if the user says something ambiguous mid-collection. Conversation history is kept as a rolling window of the last 12 messages and injected into each LLM call for context.

**RAG Pipeline**
Pricing and policy data lives in `knowledge.md`. When a user asks about plans or refunds, the `rag_node` reads that file and passes it directly into the prompt. The model is instructed to answer only from that content, which keeps responses accurate and grounded.

---

## Graph Overview

```
User Input
    |
    v
[intent_node]  -- classifies: greeting / pricing / high_intent / general
    |
    |-- greeting   --> [greeting_node]  --> END
    |-- pricing    --> [rag_node]       --> END
    |-- general    --> [general_node]   --> END
    |-- high_intent or mid-collection
                   --> [extract_node]   -- pulls name, email, platform from input
                             |
                             v
                        [lead_node]     -- asks for missing fields one at a time
                             |
                    .--------+--------.
                    |                 |
               incomplete          complete
                    |                 |
                   END          [tool_node] --> mock_lead_capture() --> END
```

---

## WhatsApp Deployment via Webhooks

To move this agent from terminal to WhatsApp, you would use the **WhatsApp Business Cloud API** from Meta and expose the agent behind a webhook.

**1. Set up a Meta Developer App**

Create an app at [developers.facebook.com](https://developers.facebook.com), add WhatsApp as a product, and get a phone number and access token.

**2. Build a webhook server**

Wrap the LangGraph agent in a FastAPI endpoint that receives incoming messages:

```python
from fastapi import FastAPI, Request
from main import graph

app = FastAPI()
sessions = {}  # use Redis in production for multi-user support

@app.post("/webhook")
async def receive_message(request: Request):
    data = await request.json()
    message = data["entry"][0]["changes"][0]["value"]["messages"][0]
    phone = message["from"]
    text = message["text"]["body"]

    state = sessions.get(phone, {
        "user_input": "", "intent": None, "response": None,
        "name": None, "email": None, "platform": None,
        "lead_collected": False, "collecting_lead": False, "history": []
    })

    state["user_input"] = text
    state["history"].append(f"User: {text}")
    state = graph.invoke(state)
    sessions[phone] = state

    send_whatsapp_reply(phone, state["response"])
    return {"status": "ok"}

@app.get("/webhook")
async def verify(request: Request):
    params = dict(request.query_params)
    if params.get("hub.verify_token") == "YOUR_VERIFY_TOKEN":
        return int(params["hub.challenge"])
```

**3. Deploy and register**

Host the server on Railway or Render with a public HTTPS URL, then paste that URL into the Meta Developer dashboard as your webhook endpoint. From that point, every WhatsApp message the user sends hits your server, runs through the agent, and gets a reply — with full state preserved across turns per phone number.

---

## For Developers

* Add or change routing logic in the `route_intent` function inside `main.py`
* Extend the knowledge base by editing `knowledge.md` — no code changes needed
* Swap in a different Gemini model by updating the model string in `call_llm()`
* Replace `mock_lead_capture()` with a real CRM API call (HubSpot, Salesforce, etc.) to go live

---

## License

MIT — free to use and modify.