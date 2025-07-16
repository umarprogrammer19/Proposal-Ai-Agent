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
    "Enter your details and preferences to find a compatible rishta. We'll send the match details to your WhatsApp!"
)

# Input form with custom prompt
with st.form("rishta_form"):
    st.markdown("### Your Details")
    name = st.text_input("Your Name", placeholder="e.g., Ali Khan")
    age = st.number_input("Your Age", min_value=18, max_value=100, step=1)
    gender = st.selectbox("Your Gender", ["Male", "Female"])
    profession = st.text_input("Your Profession", placeholder="e.g., Student")
    education = st.text_input("Your Education", placeholder="e.g., Bachelor's")
    location = st.selectbox(
        "Your Location",
        ["Karachi", "Lahore", "Islamabad", "Rawalpindi", "Faisalabad", "Peshawar"],
    )
    number = st.text_input(
        "WhatsApp Number (10 digits, no +92)",
        max_chars=10,
        placeholder="e.g., 3001234567",
    )
    st.markdown("### Match Preferences")
    custom_prompt = st.text_area(
        "Custom Match Preferences",
        placeholder="e.g., Find your perfect match with respect, compatibility, and trust. Her age is greater than me and I need an AI Engineer from Islamabad.",
    )
    submit_button = st.form_submit_button("Find Match & Send to WhatsApp")

# User data
user_data = {
    "name": name,
    "age": age,
    "gender": gender,
    "profession": profession,
    "education": education,
    "location": location,
    "number": number,
    "custom_prompt": custom_prompt,
}


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
    instructions="""You are Rishta Bot, an advanced matchmaking assistant designed to find the best match for the user based on their custom prompt and provided details.

    - Strictly interpret and enforce the user's custom prompt (e.g., 'Her age is greater than me and I need an AI Engineer from Islamabad').
    - Only select matches from the opposite gender.
    - Extract specific criteria from the custom prompt, such as:
      * Age preferences (e.g., older, younger, same age)
      * Specific profession (e.g., AI Engineer)
      * Specific location (e.g., Islamabad)
    - If the prompt specifies a profession, match it exactly (case-insensitive).
    - If the prompt specifies a location, match it exactly unless 'any location' is mentioned.
    - If the prompt specifies an age preference, filter matches accordingly.
    - If no match meets ALL criteria in the custom prompt, return: 'No match found in the data. Try adjusting your preferences.'
    - When a match is found, construct a WhatsApp message with:
      * User's details (name, age, gender, profession, education, location)
      * Match's details (name, age, profession, education, location)
      * Reasoning for the match
    - Use the `send_whatsapp_message` tool to send the message.
    - Confirm the message was sent with: 'Message successfully sent to WhatsApp.'
    """,
    tools=[send_whatsapp_message],
)


# Main logic with strict prompt-based matching
async def main(user_data):
    opposite_gender = "Female" if user_data["gender"] == "Male" else "Male"

    # Format all matches for the agent
    all_matches = [r for r in rishtas if r["gender"] == opposite_gender]

    matches_str = (
        "\n".join(
            [
                f"Name: {r['name']}, Age: {r['age']}, Profession: {r['profession']}, Education: {r['education']}, Location: {r['location']}"
                for r in all_matches
            ]
        )
        if all_matches
        else "No matches available."
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
Custom Prompt: {user_data['custom_prompt'] if user_data['custom_prompt'] else 'No specific preferences provided'}

Available Matches (opposite gender):
{matches_str}

Your task is to:
1. Interpret the custom prompt and extract specific criteria (e.g., age, profession, location).
2. Select the best match that satisfies ALL criteria in the custom prompt.
3. If no match meets all criteria, return: 'No match found in the data. Try adjusting your preferences.'
4. For a valid match, construct a WhatsApp message with user details, match details, and reasoning.
5. Send the message using the `send_whatsapp_message` tool.
6. Confirm the message was sent.
"""

    result = await Runner.run(agent, prompt, run_config=config)
    return result.final_output


# Process form submission
if submit_button:
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
            if "No match found" in reasoning:
                st.warning(reasoning)
            else:
                st.success("✅ Message sent to WhatsApp!")
                st.markdown("### 🧠 Agent Reasoning:")
                st.write(reasoning)
                st.markdown("### 📝 Your Info:")
                st.json(user_data)
