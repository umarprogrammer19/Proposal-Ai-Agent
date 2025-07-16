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
    name = st.text_input("Your Name", placeholder="Your Name")
    age = st.number_input("Your Age", min_value=18, max_value=100, step=1)
    gender = st.selectbox("Your Gender", ["Male", "Female"])
    profession = st.text_input("Your Profession", placeholder="e.g., Developer")
    education = st.text_input("Your Education", placeholder="e.g., BSCS")
    location = st.selectbox(
        "Your Location",
        ["Karachi", "Lahore", "Islamabad", "Rawalpindi", "Faisalabad", "Peshawar"],
    )
    number = st.text_input(
        "WhatsApp Number with country code without +",
        max_chars=18,
        placeholder="e.g., 923121234567",
    )
    st.markdown("### Match Preferences")
    custom_prompt = st.text_area(
        "Custom Match Preferences",
        placeholder="e.g., I want a partner older than me, Software Engineer from Karachi",
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
        "to": f"+{user_data['number']}",
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

    - Strictly interpret and enforce the user's custom prompt (e.g., 'I want a partner older than me, AI Engineer from Islamabad').
    - Only select matches from the opposite gender.
    - Extract specific criteria from the custom prompt, such as:
      * Age preferences (e.g., older, younger, same age)
      * Specific profession (e.g., AI Engineer)
      * Specific location (e.g., Islamabad)
    - If the prompt specifies a profession, match it exactly (case-insensitive).
    - If the prompt specifies a location, match it exactly unless 'any location' is mentioned.
    - If the prompt specifies an age preference, filter matches accordingly.
    - If no match meets ALL criteria in the custom prompt, return: 'No match found in the data. Try adjusting your preferences.'
    - Do NOT include the list of potential matches in the output or reasoning.
    - When a match is found, construct a WhatsApp message with:
      * User's details (name, age, gender, profession, education, location)
      * Match's details (name, age, profession, education, location)
      * Brief reasoning for the match (e.g., 'This match was chosen because...')
    - Use the send_whatsapp_message tool to send the message.
    - Confirm the message was sent with: 'Message successfully sent to WhatsApp.'
    """,
    tools=[send_whatsapp_message],
)


# Main logic with strict prompt-based matching
async def main(user_data):
    opposite_gender = "Female" if user_data["gender"] == "Male" else "Male"

    # Filter matches based on gender
    all_matches = [r for r in rishtas if r["gender"] == opposite_gender]

    # Check if the user has given a custom age preference and filter matches based on that
    min_age_diff = 0
    max_age_diff = 4

    filtered_matches = []

    for match in all_matches:
        # Age difference logic
        age_diff = abs(user_data["age"] - match["age"])

        # If the user has specified no specific preferences (empty prompt) then only filter by age and gender
        if user_data["custom_prompt"]:
            # Apply additional custom prompt filtering logic here (location, profession, etc.)
            if (
                user_data["custom_prompt"].lower() in match["profession"].lower()
                or not user_data["custom_prompt"]
            ) and min_age_diff <= age_diff <= max_age_diff:
                filtered_matches.append(match)
        else:
            if min_age_diff <= age_diff <= max_age_diff:
                filtered_matches.append(match)

    # If no valid matches, return a message to the user
    if not filtered_matches:
        return "No match found in the data. Try adjusting your preferences."

    # Choose the best match (assuming the first match is the best for simplicity)
    best_match = filtered_matches[0]

    # Construct the message with the best match details
    message = f"**Your Match Found!**\n\n"
    message += f"Name: {best_match['name']}\n"
    message += f"Age: {best_match['age']}\n"
    message += f"Profession: {best_match['profession']}\n"
    message += f"Education: {best_match['education']}\n"
    message += f"Location: {best_match['location']}\n\n"
    message += f"**Your Details:**\n"
    message += f"Name: {user_data['name']}\n"
    message += f"Age: {user_data['age']}\n"
    message += f"Profession: {user_data['profession']}\n"
    message += f"Education: {user_data['education']}\n"
    message += f"Location: {user_data['location']}\n\n"
    message += f"This match was chosen based on a {age_diff} year age difference and your custom preferences (if any)."

    # Use the send_whatsapp_message tool via the agent
    response = await Runner.run(agent, {"message": message}, run_config=config)

    return (
        response.final_output
    )  # The agent should handle sending the message automatically


# Process form submission
if submit_button:
    if not api or not token:
        st.error("API Key or Token is missing.")
    elif not all([name, age, gender, profession, education, location, number]):
        st.warning("Please fill in all required fields.")
    else:
        number = number.replace(" ", "").replace("-", "")
        if len(number) > 18 or not number.isdigit():
            st.error("Enter a valid WhatsApp number.")
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
