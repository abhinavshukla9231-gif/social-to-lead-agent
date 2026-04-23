
from typing import TypedDict, Optional, List
from langgraph.graph import StateGraph
from google import genai
import os
import re
from dotenv import load_dotenv

# ---------------- LOAD ENV ----------------
load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


# ---------------- STATE ----------------
class AgentState(TypedDict):
    user_input: str
    intent: Optional[str]
    response: Optional[str]

    name: Optional[str]
    email: Optional[str]
    platform: Optional[str]

    lead_collected: bool
    collecting_lead: bool

    history: List[str]


# ---------------- TOOL ----------------
def mock_lead_capture(name: str, email: str, platform: str):
    """Mock API to simulate saving a lead to a CRM/database."""
    print(f"\nLead captured successfully: {name}, {email}, {platform}\n")


# ---------------- HELPER ----------------
def call_llm(prompt: str) -> str:
    """Call Gemini 2.5 Flash-Lite — free tier: 1,000 req/day, 15 req/min."""
    import time
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=prompt,
            )
            return response.text.strip()
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                print("Rate limit reached. Waiting 60 seconds before retrying...")
                time.sleep(60)
            else:
                raise
    raise RuntimeError("API rate limit exceeded after all retries.")


def load_knowledge_base() -> str:
    """Load the local knowledge base markdown file."""
    kb_path = os.path.join(os.path.dirname(__file__), "knowledge.md")
    with open(kb_path, "r") as f:
        return f.read()


# ---------------- NODES ----------------

def detect_intent(state: AgentState) -> AgentState:
    """
    Node 1: Classify the user's latest message into one of four intents.
    Uses conversation history for better context.
    """
    history_str = "\n".join(state["history"][-6:]) if state["history"] else "None"

    prompt = f"""You are an intent classifier for AutoStream, a SaaS video editing tool.

Conversation History:
{history_str}

Latest User Message: {state["user_input"]}

Classify the user's intent into EXACTLY one of these categories:
- greeting     → simple hello, hi, hey, or casual opener
- pricing      → asking about plans, features, cost, refunds, or policies
- high_intent  → ready to sign up, subscribe, try, buy, or get started
- general      → anything else

Rules:
- If the user mentions wanting to try, subscribe, sign up, get started, or buy → high_intent
- If the user asks about price, cost, plans, features, refunds, support → pricing
- Reply with ONLY the single word label, nothing else.
"""
    raw = call_llm(prompt).lower().strip()

    # Sanitize: extract only the valid intent word
    valid_intents = ["greeting", "pricing", "high_intent", "general"]
    intent = "general"
    for vi in valid_intents:
        if vi in raw:
            intent = vi
            break

    return {**state, "intent": intent}


def greeting_node(state: AgentState) -> AgentState:
    """
    Node 2: Handle casual greetings with a warm welcome message.
    """
    return {
        **state,
        "response": (
            "Welcome to AutoStream — your AI-powered video editing platform for content creators.\n\n"
            "I can help you with:\n"
            "- Pricing and Plans\n"
            "- Features and Policies\n"
            "- Getting started\n\n"
            "What would you like to know?"
        ),
    }


def rag_node(state: AgentState) -> AgentState:
    """
    Node 3: Answer product/pricing questions using the local knowledge base (RAG).
    """
    context = load_knowledge_base()
    history_str = "\n".join(state["history"][-6:]) if state["history"] else "None"

    prompt = f"""You are a helpful assistant for AutoStream, a SaaS video editing platform.

Answer the user's question using ONLY the information provided in the Context below.
If the answer is not in the context, say: "I don't have that information, but feel free to contact our support team."

Do NOT use emojis. Preserve the structure of the knowledge base exactly — use bullet points and line breaks as they appear in the source.
Do NOT make up any information.

Conversation History:
{history_str}

Context (Knowledge Base):
{context}

User Question: {state["user_input"]}

Provide a clear and concise answer:
"""
    response = call_llm(prompt)
    return {**state, "response": response}


def extract_details(state: AgentState) -> AgentState:
    """
    Node 4: Extract name, email, and platform from the user's message.
    Preserves already-collected values (does not overwrite).
    Uses regex for reliable email extraction and keyword match for platform.
    Falls back to simple heuristics for name detection.
    """
    text = state["user_input"].strip()

    # Preserve existing values — never overwrite collected data
    name = state.get("name")
    email = state.get("email")
    platform = state.get("platform")

    # --- EMAIL: regex-based (most reliable) ---
    if not email:
        match = re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text)
        if match:
            email = match.group()

    # --- PLATFORM: keyword matching ---
    if not platform:
        known_platforms = ["youtube", "instagram", "tiktok", "twitter", "facebook", "linkedin"]
        for p in known_platforms:
            if p in text.lower():
                platform = p.capitalize()
                break

    # --- NAME: heuristic (short alpha string, no email, not a stop word) ---
    if not name:
        stop_words = {
            "yes", "yeah", "yup", "ok", "okay", "sure", "hi", "hello", "hey",
            "please", "thanks", "thank", "no", "nope", "subscribe", "pro",
            "basic", "plan", "email", "name", "platform",
        }
        words = text.split()
        # Accept 1–3 word inputs that are alphabetic and not stop words
        if (
            "@" not in text
            and 1 <= len(words) <= 3
            and all(w.replace(".", "").replace("-", "").isalpha() for w in words)
            and text.lower() not in stop_words
            and not any(w.lower() in stop_words for w in words)
        ):
            name = text.strip().title()

    return {
        **state,
        "name": name,
        "email": email,
        "platform": platform,
    }


def lead_node(state: AgentState) -> AgentState:
    """
    Node 5: Collect lead details one field at a time.
    Asks for name → email → platform sequentially.
    Only marks lead_collected=True when all three are present.
    """
    # Step 1: Ask for name
    if not state.get("name"):
        return {
            **state,
            "collecting_lead": True,
            "response": "I would like to get you set up. Could you share your name?",
        }

    # Step 2: Ask for email
    if not state.get("email"):
        return {
            **state,
            "collecting_lead": True,
            "response": f"Thank you, {state['name']}. What is your email address?",
        }

    # Step 3: Ask for platform
    if not state.get("platform"):
        return {
            **state,
            "collecting_lead": True,
            "response": "Which platform do you create content on? (e.g., YouTube, Instagram, TikTok)",
        }

    # All collected → mark lead as complete
    return {
        **state,
        "lead_collected": True,
        "collecting_lead": False,
        "response": None,  # tool_node will set the final response
    }


def tool_node(state: AgentState) -> AgentState:
    """
    Node 6: Execute the mock lead capture tool.
    Only called after all three fields are confirmed present.
    """
    mock_lead_capture(
        name=state["name"],
        email=state["email"],
        platform=state["platform"],
    )

    return {
        **state,
        "response": (
            f"You are all set, {state['name']}. "
            f"We have captured your details and our team will reach out to your {state['platform']} account shortly. "
            "Welcome to AutoStream."
        ),
    }


def general_node(state: AgentState) -> AgentState:
    """
    Fallback node for general queries not matched by other intents.
    """
    history_str = "\n".join(state["history"][-6:]) if state["history"] else "None"

    prompt = f"""You are a helpful assistant for AutoStream, a SaaS video editing platform for content creators.

Do NOT use emojis or markdown formatting. Respond in plain, professional text only.

Conversation History:
{history_str}

User said: {state["user_input"]}

Respond helpfully. If you don't know the answer, suggest they ask about pricing or getting started.
"""
    response = call_llm(prompt)
    return {**state, "response": response}


# ---------------- ROUTERS ----------------

def route_intent(state: AgentState) -> str:
    """
    Router after intent detection.

    Priority order:
    1. If lead collection is already in progress → continue collecting
    2. If high_intent → start lead flow
    3. Otherwise route by intent, never entering lead flow
    """
    # Priority 1: Continue an already-started lead flow
    if not state.get("lead_collected") and state.get("collecting_lead") is True:
        return "extract"

    # Priority 2: Start lead flow only on explicit high intent
    if state["intent"] == "high_intent":
        return "extract"

    # Priority 3: All other intents go to their own nodes — never lead flow
    intent_map = {
        "greeting": "greeting",
        "pricing": "rag",
        "general": "general",
    }
    return intent_map.get(state["intent"], "general")


def check_lead_complete(state: AgentState) -> str:
    """
    Router after lead_node.
    If all data is collected → trigger the tool.
    Otherwise → end (lead_node already asked a follow-up question).
    """
    if state.get("lead_collected"):
        return "tool"
    return "__end__"


# ---------------- BUILD GRAPH ----------------

builder = StateGraph(AgentState)

# Register nodes
builder.add_node("intent", detect_intent)
builder.add_node("greeting", greeting_node)
builder.add_node("rag", rag_node)
builder.add_node("extract", extract_details)
builder.add_node("lead", lead_node)
builder.add_node("tool", tool_node)
builder.add_node("general", general_node)

# Entry point
builder.set_entry_point("intent")

# Conditional routing after intent detection
builder.add_conditional_edges(
    "intent",
    route_intent,
    {
        "greeting": "greeting",
        "rag": "rag",
        "extract": "extract",
        "general": "general",
    },
)

# After extraction → always go to lead collection check
builder.add_edge("extract", "lead")

# After lead node → check if complete
builder.add_conditional_edges(
    "lead",
    check_lead_complete,
    {
        "tool": "tool",
        "__end__": "__end__",
    },
)

# Terminal edges
builder.add_edge("tool", "__end__")
builder.add_edge("rag", "__end__")
builder.add_edge("greeting", "__end__")
builder.add_edge("general", "__end__")

graph = builder.compile()


# ---------------- CHAT LOOP ----------------

def run_chat():
    """Main conversational loop with persistent state across turns."""
    state: AgentState = {
        "user_input": "",
        "intent": None,
        "response": None,
        "name": None,
        "email": None,
        "platform": None,
        "lead_collected": False,
        "collecting_lead": False,
        "history": [],
    }

    print("=" * 50)
    print("AutoStream Assistant (LangGraph + Gemini)")
    print("Type 'exit' to quit.")
    print("=" * 50 + "\n")

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit", "bye"):
            print("Bot: Thanks for chatting. Have a great day.")
            break

        # Inject user input into state
        state["user_input"] = user_input
        state["history"].append(f"User: {user_input}")

        # Run the graph
        state = graph.invoke(state)

        # Print bot response
        bot_response = state.get("response") or "I'm sorry, I didn't understand that. Could you rephrase?"
        print(f"Bot: {bot_response}\n")

        # Append to history (keep last 12 messages = ~6 turns)
        state["history"].append(f"Bot: {bot_response}")
        state["history"] = state["history"][-12:]


if __name__ == "__main__":
    run_chat()