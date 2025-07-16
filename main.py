import os
import asyncio
import requests
from dotenv import load_dotenv
import streamlit as st
from agents import (
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
    RunConfig,
    Agent,
    Runner,
    function_tool,
)
from data import rishtas

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


# Agent setup with updated instructions
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
      * Age preferences (e.g., older, younger, same age, specific age)
      * Specific profession (e.g., AI Engineer)
      * Specific location (e.g., Islamabad)
    - For age:
      * If the prompt specifies an age preference, filter matches accordingly.
      * If no age preference is specified, assume the user wants a match within 3 years of their own age.
    - For location:
      * If the prompt specifies a location, match it exactly unless 'any location' is mentioned.
      * If no location is specified, prefer matches from the same location as the user.
    - For profession:
      * If the prompt specifies a profession, match it exactly (case-insensitive).
      * If no profession is specified, use the user's provided profession for filtering, or if that's also not available, do not apply a profession filter.
    - Select only matches that satisfy ALL specified criteria and default filters where applicable.
    - If no match meets all the criteria, return: 'No match found in the data. Try adjusting your preferences.'
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
    user_age = user_data["age"]

    # --- Pre-filtering the rishtas data ---
    # Filter by opposite gender and default age range (3 years difference)
    pre_filtered_matches = []
    for r in rishtas:
        if r["gender"] == opposite_gender:
            if abs(r["age"] - user_age) <= 4:
                pre_filtered_matches.append(r)

    # Now, the agent will process this pre_filtered_matches list
    matches_str = (
        "\n".join(
            [
                f"Name: {r['name']}, Age: {r['age']}, Profession: {r['profession']}, Education: {r['education']}, Location: {r['location']}"
                for r in pre_filtered_matches
            ]
        )
        if pre_filtered_matches
        else "No suitable initial matches found based on gender and general age range."
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

Available Matches (opposite gender, pre-filtered by a general age range (user_age +/- 4 years) and gender):
{matches_str}

Your task is to:
1. Carefully interpret the 'Custom Prompt' to extract **strict** criteria for age, profession, and location.
2. If the 'Custom Prompt' specifies an **age preference**, override the default age range and apply it strictly (e.g., "older than me", "exactly 25"). If no age preference is in the custom prompt, apply a strict age filter of **+/- 3 years** from the user's age.
3. If the 'Custom Prompt' specifies a **profession**, match it exactly (case-insensitive). If no profession is specified in the custom prompt, *and* the user provided their own profession, prioritize finding a match with a similar profession. If neither is specified, do not filter by profession.
4. If the 'Custom Prompt' specifies a **location**, match it exactly. If no location is specified in the custom prompt, prefer matches from the user's same location.
5. From the "Available Matches" list, **select only ONE best match** that satisfies ALL criteria derived from the custom prompt and default rules. Prioritize exact matches for custom prompt criteria.
6. If no match meets ALL the criteria, return: 'No match found in the data. Try adjusting your preferences.'
7. Do NOT include the list of potential matches in the output or reasoning.
8. For a valid match, construct a WhatsApp message with:
    * User's details (name, age, gender, profession, education, location)
    * Match's details (name, age, profession, education, location)
    * A brief, clear reasoning for the match, explicitly stating how the match meets the user's preferences (e.g., "This match was chosen because [Match Name] is a [Match Profession] from [Match Location], which aligns with your preference for a [User Profession] and [User Location], and is within your preferred age range.").
9. Send the message using the send_whatsapp_message tool.
10. Confirm the message was sent with: 'Message successfully sent to WhatsApp.'
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
