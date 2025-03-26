import streamlit as st
from streamlit_chat import message
import google.generativeai as genai
import os
import tempfile
import time
import speech_recognition as sr
from gtts import gTTS
from audio_recorder_streamlit import audio_recorder
from datetime import datetime
import pandas as pd
import re
import glob


class GeminiBot:
    """Define the class for the Gemini Chatbot."""

    def __init__(self, api_key: str):
        """Initialize a new instance of the GeminiBot class.
        Args:
        - api_key (string): The Gemini API key used to authenticate
        with the Google Generative AI service.
        """
        # Configure API
        self.api_key = api_key
        genai.configure(api_key=self.api_key)
        
        # Initialize the message history for chat storing
        if "message_history" not in st.session_state:
            st.session_state["message_history"] = []

    def respond(self, user_message: str, model_name: str) -> str:
        """Method to send user's message to Gemini model and receive API
        response. This method also documents and updates the message history
        between the user and the bot.
        Args:
        - user_message (string): The user's input message.
        - model_name (string): The Gemini model to use.
        Returns:
        - str: Bot's response message.
        """
        try:
            # Append user message to history
            request = {"role": "user", "content": user_message}
            st.session_state["message_history"].append(request)
            
            # Create a generative model using Gemini API
            model = genai.GenerativeModel(model_name)
            
            # Get response from the model
            response = model.generate_content(user_message)
            bot_message = response.text if response else "No response received."
            
            # Append bot response to history
            bot_response = {"role": "assistant", "content": bot_message}
            st.session_state["message_history"].append(bot_response)
            
            return bot_message
        except Exception as e:
            return f"Error: {str(e)}"

    def say(self, bot_message: str):
        """Method to convert bot's message into speech audio.
        Args:
        - bot_message (string): The bot's text message to convert.
        """
        # Convert bot's message from text to speech
        tts = gTTS(text=bot_message, lang='en')
        bot_audio_filename = f"bot-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.mp3"
        tts.save(bot_audio_filename)
        
        # Read the bot audio file
        with open(bot_audio_filename, "rb") as bot_audio_file:
            bot_audio_bytes = bot_audio_file.read()
        
        # Display an audio button on the page that plays the bot audio file
        st.write("Play the audio below to LISTEN to the bot")
        st.audio(bot_audio_bytes, format="audio/mp3")

    def chat(self, user_message: str, text_or_speak: str, selected_model: str):
        """Method to respond to user's message in both text and speech audio.
        Args:
        - user_message (string): The user's input message.
        - text_or_speak (string): Type of communication.
        - selected_model (string): The Gemini model to use.
        """
        if user_message.strip():
            # Send user message to Gemini model and get bot's message
            bot_message = self.respond(
                user_message=user_message, model_name=selected_model
            )
            # Save the user message in streamlit session state
            st.session_state[
                f"user-{text_or_speak}"
            ].append(user_message)
            # Save the bot's message in streamlit session state
            st.session_state[
                f"bot-{text_or_speak}"
            ].append(bot_message)

            # Play the latest bot's message in audio
            self.say(bot_message)

    def transcribe_voice(self, audio_bytes: bytes) -> str:
        """Method to transcribe user's input speech to text.
        Args:
        - audio_bytes (bytes): The audio data in bytes to be transcribed.
        Returns:
        - str: The transcribed text from the audio input.
        """
        # Write audio_bytes into a new WAV file
        temp_dir = tempfile.mkdtemp()
        filename = os.path.join(temp_dir, f"user-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.wav")
        with open(filename, "wb") as f:
            f.write(audio_bytes)
        
        # Transcribe audio file to text
        recognizer = sr.Recognizer()
        with sr.AudioFile(filename) as source:
            audio_data = recognizer.record(source)
        try:
            return recognizer.recognize_google(audio_data)
        except sr.UnknownValueError:
            return "Could not understand audio"
        except sr.RequestError:
            return "Speech Recognition API error"


class ChatApp:
    """Define the class for the Chat Application"""

    def __init__(self):
        """Initialize a new instance of the ChatApp class."""
        # Set page config
        st.set_page_config(
            page_title="Talk To Gemini",
            page_icon=":robot_face:",
            layout="centered",
            initial_sidebar_state="auto",
        )
        # Remove all previously existing audio files
        for file in glob.glob("./*.mp3") + glob.glob("./*.wav"):
            try:
                os.remove(file)
            except:
                pass
                
        # Initialize session state variables for chat storing
        if "bot-text" not in st.session_state:
            st.session_state["bot-text"] = []
        if "user-text" not in st.session_state:
            st.session_state["user-text"] = []
        if "bot-speak" not in st.session_state:
            st.session_state["bot-speak"] = []
        if "user-speak" not in st.session_state:
            st.session_state["user-speak"] = []
        if "prompts" not in st.session_state:
            try:
                # Try Load a series of role-based prompts from an online CSV file
                df = pd.read_csv(
                    "https://raw.githubusercontent.com/f/"
                    "awesome-chatgpt-prompts/main/prompts.csv"
                )
                # Update prompt column
                df["prompt"] = df["prompt"].apply(
                    lambda x: self.transform_prompt(x)
                )
                # Save the prompts in session state
                st.session_state["prompts"] = df
            except:
                # If prompt loading fails, display an error message
                st.error(
                    "Unable to load the built-in prompts. Please check "
                    "[awesome-chatgpt-prompts](https://github.com/f/awesome-"
                    "chatgpt-prompts/blob/main/prompts.csv) for more details."
                )

    # Transform a prompt to appropriate format
    def transform_prompt(self, x):
        # Add full stop to the end of each prompt
        if x[-1] != ".":
            x = x + "."
        # Find the sentences within the prompt that contain 'My first...'
        list_my_first = re.findall(r". my first [^.]+.", x.lower())
        # If the pattern was found in the prompt...
        if list_my_first:
            my_first = list_my_first[-1]
            cutoff_id = x.lower().index(my_first)
            # remove the last 'My first...' sentence from the prompt
            prompt = x[: cutoff_id + 1]
        else:
            prompt = x
        # Add "Reply "OK" to confirm." to the end of each prompt
        if not prompt.endswith("""Reply "OK" to confirm."""):
            prompt = prompt + """ Reply "OK" to confirm."""
        return prompt

    # Display chat history as conversation dialogs
    def output_chat_history(self, text_or_speak):
        # Check if there is any chat history for the specified conversation type
        if st.session_state[f"bot-{text_or_speak}"]:
            # Get the list of historical bot's messages
            list_bot_messages = st.session_state[
                f"bot-{text_or_speak}"
            ]
            # Get the list of historical user's messages
            list_user_messages = st.session_state[
                f"user-{text_or_speak}"
            ]
            # Iterate through the chat history in reverse order
            for i in range(len(list_bot_messages) - 1, -1, -1):
                # Display the bot's message first
                message(
                    list_bot_messages[i],
                    is_user=False,
                    avatar_style="bottts-neutral",
                    seed=75,
                    key=f"bot-{text_or_speak}-{i}",
                )
                # Display the user's message right after bot's message
                message(
                    list_user_messages[i],
                    is_user=True,
                    avatar_style="adventurer-neutral",
                    seed=124,
                    key=f"user-{text_or_speak}-{i}",
                )

    # Run the Chatbot application
    def run(self):
        # Set the page title
        st.title("Welcome to Talk To Gemini")
        # Display a subheader that briefly describe the chatbot web app
        st.subheader(
            "Empowering Conversations: A ChatBot You Can Message Or Talk "
            "To, Powered By Google's Gemini AI Models and Speech Technologies "
            ":robot_face:"
        )

        col1, col2, col3 = st.columns([1, 0.2, 1])
        # Get the Gemini model selected by the user
        MODEL = col1.selectbox(
            "Select a Gemini model",
            ("models/gemini-1.5-pro", "models/gemini-1.5-flash", "models/gemini-pro"),
            help=(
                "Gemini 1.5 Pro is the recommended model for complex reasoning and "
                "conversational tasks. Gemini 1.5 Flash is faster and more cost-effective "
                "for everyday tasks. Gemini Pro is the previous generation model."
            ),
        )
        # Get the API key from the user
        KEY = col3.text_input(
            "Enter your API Key",
            type="password",
            value="AIzaSyDOxeW2OM5Pw9U88t6783h7UBmf0yf2NIg",  # Default key from your original code
            help=(
                "To create and collect a Google AI API key, visit "
                "https://ai.google.dev/ and click on 'Get API key'."
            ),
        )

        # Mark down a breakline
        st.markdown("***")

        # If API key is entered
        if KEY.strip():
            # Initialize the chatbot with the provided API key
            bot = GeminiBot(KEY)

            # Mark down a pro tip for users
            st.markdown(
                """*Pro tip: If you wish to initiate a new conversation, you
                can either refresh the webpage, choose "[Clear conversation
                history]" from the built-in prompt dropdown below, or enter
                the command "Ignore all previous instructions before this
                one".*"""
            )

            # Display an empty line
            st.text("")
            # Two Expanders for communication with the bot
            # Expander 1: Message to bot
            with st.expander(":memo: MESSAGE BOT"):
                if "prompts" in st.session_state:
                    df_prompts = st.session_state["prompts"]
                    # Get list of roles for prompts
                    prompts = sorted(list(df_prompts["act"]))
                    # Add options to the prompt list
                    prompts = tuple(
                        [
                            "You want the bot to act as...",
                            "[Clear conversation history]"
                        ] + prompts
                    )
                    # Dropdown box for built-in prompt selection
                    prompt_act_selected = st.selectbox(
                        label="Choose a built-in prompt (optional)",
                        options=prompts,
                        index=0,
                        help=(
                            "The collection of built-in prompts were imported "
                            "from [awesome-chatgpt-prompts]"
                            "(https://github.com/f/awesome-chatgpt-prompts)."
                        ),
                    )
                    # Set the initial value for text message field based on the selected prompt
                    if prompt_act_selected == "You want the bot to act as...":
                        initial_value = ""
                    elif prompt_act_selected == "[Clear conversation history]":
                        initial_value = (
                            "Ignore all previous instructions before this one."
                        )
                        # Clear the message history
                        st.session_state["message_history"] = []
                    else:
                        prompt_id = list(
                            df_prompts[df_prompts.act == prompt_act_selected].index
                        )[0]
                        initial_value = df_prompts.loc[prompt_id, "prompt"]
                else:
                    initial_value = ""
                
                # Text message input field with initial value
                user_message_text = st.text_area(
                    "Send text message",
                    placeholder=(
                        "Type your text message here and press Ctrl+Enter "
                        "to submit"
                    ),
                    value=initial_value,
                    height=120,
                )
                
                # Button to send message
                if st.button("Send Message"):
                    # Send user's text message to the bot
                    bot.chat(
                        user_message=user_message_text,
                        text_or_speak="text",
                        selected_model=MODEL,
                    )
                
                # Output chat history
                st.text("")
                self.output_chat_history("text")
                st.text("")

            # Expander 2: Talk to bot
            with st.expander(":speaking_head_in_silhouette: TALK TO BOT"):
                # Check if the text message field entry contains any text
                if user_message_text.strip() and user_message_text != initial_value:
                    # If so, display an error message
                    st.error(
                        "Please ensure that all content has been deleted "
                        "from the text message field"
                    )
                else:
                    # If text entry is empty, show Audio recorder
                    st.write("Click below to record your voice (stops after 3s pause)")
                    audio_bytes = audio_recorder(
                        neutral_color="#eeeeee", pause_threshold=3.0
                    )
                    if audio_bytes:
                        # Transcribe the user's voice
                        user_message_voice = bot.transcribe_voice(
                            audio_bytes=audio_bytes
                        )
                        # Display a status message with the transcribed text
                        st.success(
                            f"Voice recording finished. You said: {user_message_voice}"
                        )
                        # Send user's voice message to the bot
                        bot.chat(
                            user_message=user_message_voice,
                            text_or_speak="speak",
                            selected_model=MODEL,
                        )
                    
                    # Alternative: File uploader for audio
                    st.write("OR upload an audio file:")
                    uploaded_audio = st.file_uploader("Upload an audio file:", type=["wav", "mp3", "ogg"])
                    if uploaded_audio:
                        temp_dir = tempfile.mkdtemp()
                        temp_audio_path = os.path.join(temp_dir, uploaded_audio.name)
                        with open(temp_audio_path, "wb") as f:
                            f.write(uploaded_audio.read())
                        
                        user_message = bot.transcribe_voice(open(temp_audio_path, "rb").read())
                        st.markdown(
                            f"""
                            <div style='background-color:#E3F2FD; padding:10px; border-radius:10px; margin:5px 0; display:flex; align-items:center; border: 1px solid #90CAF9;'>
                                <span style='margin-right:10px;'>🗣️</span> <b>You said:</b> <span style='color:#0D47A1;'>{user_message}</span>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        
                        if user_message:
                            bot.chat(
                                user_message=user_message,
                                text_or_speak="speak",
                                selected_model=MODEL,
                            )
                    
                    # Output chat history
                    st.text("")
                    self.output_chat_history("speak")
                    st.text("")

        else:
            # If API key is not entered, display an error message
            st.error("Please enter your API key to initiate your chat!")


# Check if this script is being run directly
if __name__ == "__main__":
    # Create an instance of ChatApp class
    chat_app = ChatApp()
    # Call run() method to start the chat application
    chat_app.run()