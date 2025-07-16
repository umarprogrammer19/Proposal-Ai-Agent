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

# Input form
with st.form("rishta_form"):
    name = st.text_input("Your Name", placeholder="e.g., Ali Khan")
    age = st.number_input("Your Age", min_value=18, max_value=100, step=1)
    gender = st.selectbox("Your Gender", ["Male", "Female"])
    profession = st.text_input("Your Profession", placeholder="e.g., Engineer")
    education = st.text_input("Your Education", placeholder="e.g., Bachelor's")
    number = st.text_input(
        "WhatsApp Number (10 digits, no +92)",
        max_chars=10,
        placeholder="e.g., 3001234567",
    )
    message = st.text_area(
        "Your Intro (Optional)", placeholder="Tell us about yourself..."
    )
    submit_button = st.form_submit_button("Find Match & Send to WhatsApp")

# User data
user_data = {
    "name": name,
    "age": age,
    "gender": gender,
    "profession": profession,
    "education": education,
    "number": number,
    "message": message.replace("\n", " ") if message else "",
}


# WhatsApp tool updated to accept message parameter
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


# Agent setup
external_agent = AsyncOpenAI(
    api_key=api, base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)
model = OpenAIChatCompletionsModel(
    openai_client=external_agent, model="gemini-2.0-flash"
)
config = RunConfig(model=model, model_provider=external_agent, tracing_disabled=True)

agent = Agent(
    name="Rishta_Bot",
    instructions="""You are Rishta Bot, designed to find the best match for the user from a list of potential rishtas based on compatibility.

    - Consider factors like age, profession, education, and location (if available) to assess compatibility.
    - Select the most suitable rishta from the opposite gender and explain why it’s a good match.
    - Construct a WhatsApp message with the user’s details and the selected rishta’s details in a clear format.
    - If the user provided an intro message, include it as 'Intro: [message]' in the WhatsApp message.
    - Send the message using the `send_whatsapp_message` tool with the constructed message as the argument.
    - If no suitable match is found, send a message informing the user.
    - Confirm after sending that the message was sent.""",
    tools=[send_whatsapp_message],
)


# Main logic with AI agent
async def main():
    opposite_gender = "Female" if user_data["gender"] == "Male" else "Male"
    eligible_matches = [r for r in rishtas if r["gender"] == opposite_gender]

    matches_str = (
        "\n".join(
            [
                f"{i+1}. Name: {r['name']}, Age: {r['age']}, Gender: {r['gender']}, Profession: {r['profession']}, Education: {r['education']}, Location: {r.get('location', 'Not Provided')}"
                for i, r in enumerate(eligible_matches)
            ]
        )
        if eligible_matches
        else "No matches available."
    )

    prompt = f"""
    User Details:
    Name: {user_data['name']}
    Age: {user_data['age']}
    Gender: {user_data['gender']}
    Profession: {user_data['profession']}
    Education: {user_data['education']}
    Message: {user_data['message'] if user_data['message'] else 'None'}

    Potential Matches:
    {matches_str}

    Select the best match and follow your instructions.
    """

    result = await Runner.run(agent, prompt, run_config=config)
    return result.final_output


# Execution on button click
if submit_button:
    if not api or not token:
        st.error("API Key or Token is missing.")
    elif not all([name, age, gender, profession, education, number]):
        st.warning("Please fill in all required fields.")
    else:
        number = number.replace(" ", "").replace("-", "")
        if len(number) != 10 or not number.isdigit():
            st.error("Enter a valid 10-digit WhatsApp number.")
        else:
            with st.spinner("Finding your match..."):
                reasoning = asyncio.run(main())
            st.success("✅ Message sent to WhatsApp!")
            st.markdown("### 🧠 Agent Reasoning:")
            st.write(reasoning)
            st.markdown("### 📝 Your Info:")
            st.json(user_data)
