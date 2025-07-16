import os
import asyncio
import requests
from dotenv import load_dotenv
import streamlit as st
from data import rishtas
from agents import (
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
    RunConfig,
    Agent,
    Runner,
    function_tool,
)

# Load environment variables
load_dotenv()
api = os.getenv("OPENAI_KEY")
token = os.getenv("TOKEN")
instance = os.getenv("INSTANCE")

# Custom CSS for professional UI
st.markdown(
    """
    <style>
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-size: 16px;
            border-radius: 5px;
            padding: 12px;
            width: 100%;
            margin-top: 20px;
        }
        .stTextInput>label, .stNumberInput>label, .stSelectbox>label, .stTextArea>label {
            font-weight: bold;
            color: #333;
        }
        .stTextInput input, .stNumberInput input, .stSelectbox select, .stTextArea textarea {
            border-radius: 5px;
            padding: 10px;
            border: 1px solid #ccc;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Streamlit UI
st.title("**Rishta Bot 💌 - Find Your Perfect Match!**")
st.markdown(
    "Enter your details to find a compatible rishta. We'll send the match details to your WhatsApp!"
)

# Input form with location added
with st.form("rishta_form"):
    name = st.text_input("Your Name", placeholder="e.g., Ali Khan")
    age = st.number_input("Your Age", min_value=18, max_value=100, step=1)
    gender = st.selectbox("Your Gender", ["Male", "Female"])
    profession = st.text_input("Your Profession", placeholder="e.g., Engineer")
    education = st.text_input("Your Education", placeholder="e.g., Bachelor's")
    locations = [
        "Karachi",
        "Lahore",
        "Islamabad",
        "Rawalpindi",
        "Faisalabad",
        "Peshawar",
    ]
    location = st.selectbox("Your Location", locations)
    number = st.text_input(
        "WhatsApp Number (10 digits, no +92)",
        max_chars=10,
        placeholder="e.g., 3001234567",
    )
    message = st.text_area(
        "Your Intro (Optional)", placeholder="Tell us about yourself..."
    )
    submit_button = st.form_submit_button("Find Match & Send to WhatsApp")


# WhatsApp sending tool
@function_tool
def send_whatsapp_message(message: str):
    url = f"https://api.ultramsg.com/{instance}/messages/chat"
    payload = {
        "token": token,
        "to": f"+92{user_data['number']}",
        "body": message,
    }
    res = requests.post(url, data=payload)
    return res.text


# Agent setup with enhanced instructions
external_agent = AsyncOpenAI(
    api_key=api, base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)
model = OpenAIChatCompletionsModel(
    openai_client=external_agent, model="gemini-2.0-flash"
)
config = RunConfig(model=model, model_provider=external_agent, tracing_disabled=True)

agent = Agent(
    name="Rishta_Bot",
    instructions="""You are Rishta Bot, an advanced matchmaking assistant designed to find the best possible match for the user from a list of potential rishtas based on compatibility.

    - Assess compatibility using these factors:
      * Age: Prefer matches within a 5-year age difference when possible
      * Profession: Consider similar or complementary career fields
      * Education: Look for comparable education levels
      * Location: Strongly prefer matches from the same location unless none are available
    - Select the most suitable rishta from the opposite gender
    - Provide a clear explanation of why the selected match is compatible
    - Construct a WhatsApp message with:
      * User's details
      * Selected rishta's details
      * Intro message (if provided) as 'Intro: [message]'
    - Use the `send_whatsapp_message` tool to send the message
    - If no suitable match is found, send a message stating: "Sorry, no compatible matches found at this time."
    - Confirm message sending with: "Message successfully sent to WhatsApp"
    """,
    tools=[send_whatsapp_message],
)


# Main logic with location-based matching
async def main(user_data):
    opposite_gender = "Female" if user_data["gender"] == "Male" else "Male"

    # First try matches from same location
    same_location_matches = [
        r
        for r in rishtas
        if r["gender"] == opposite_gender and r["location"] == user_data["location"]
    ]

    if same_location_matches:
        eligible_matches = same_location_matches
        location_note = "These matches are from the same location as the user."
    else:
        eligible_matches = [r for r in rishtas if r["gender"] == opposite_gender]
        location_note = "No matches found in the same location; showing matches from other locations."

    # Format eligible matches for the agent
    eligible_matches_str = "\n".join(
        [
            f"{i+1}. Name: {r['name']}, Age: {r['age']}, Profession: {r['profession']}, Education: {r['education']}, Location: {r['location']}"
            for i, r in enumerate(eligible_matches)
        ]
    )

    # Detailed prompt for the agent
    prompt = f"""
You are Rishta Bot. The user has provided the following details:

Name: {user_data['name']}
Age: {user_data['age']}
Gender: {user_data['gender']}
Profession: {user_data['profession']}
Education: {user_data['education']}
Location: {user_data['location']}
Intro message: {user_data['message'] if user_data['message'] else 'None'}

{location_note}

Here is the list of eligible matches from the opposite gender:

{eligible_matches_str}

Your task is to:
1. Select the most compatible match based on age (prefer within 5 years), profession (similar/complementary), education (comparable level), and location (same location preferred)
2. Explain why this match was chosen
3. Construct and send a WhatsApp message with both user and match details
4. Confirm the message was sent
If no suitable match exists, inform the user appropriately.
"""

    result = await Runner.run(agent, prompt, run_config=config)
    return result.final_output


# Process form submission
if submit_button:
    user_data = {
        "name": name,
        "age": age,
        "gender": gender,
        "profession": profession,
        "education": education,
        "location": location,
        "number": number,
        "message": message.replace("\n", " ") if message else "",
    }

    if not api or not token:
        st.error("API Key or Token is missing.")
    elif not all([name, age, gender, profession, education, location, number]):
        st.warning("Please fill in all required fields.")
    else:
        number = number.replace(" ", "").replace("-", "")
        if len(number) != 10 or not number.isdigit():
            st.error("Enter a valid 10-digit WhatsApp number.")
        else:
            with st.spinner("Finding your match..."):
                reasoning = asyncio.run(main(user_data))
            st.success("✅ Message sent to WhatsApp!")
            st.markdown("### 🧠 Agent Reasoning:")
            st.write(reasoning)
            st.markdown("### 📝 Your Info:")
            st.json(user_data)
