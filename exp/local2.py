import asyncio
import pyaudio
import time
import os # Need this for GEMINI_API_KEY
from google import genai
from google.genai import types

# --- Configuration ---
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000 # Send rate (input)
CHUNK = 512 # Keep this small for low latency

# 1. Define the Client and Configuration
# Assumes API key is in GEMINI_API_KEY environment variable
# If not set, you must hardcode or load it like: api_key="YOUR_API_KEY"
client = genai.Client(
    http_options={"api_version": "v1beta"},
    api_key=os.getenv("GEMINI_API_KEY"), # Added os import and lookup
)

# System instruction should be loaded here (from system_instruction.txt)
SYSTEM_INSTRUCTION_TEXT = "" 
try:
    with open("system_instruction.txt", "r") as f:
        SYSTEM_INSTRUCTION_TEXT = f.read().strip()
except FileNotFoundError:
    print("Warning: system_instruction.txt not found. Using default empty instruction.")

MODEL_NAME = "models/gemini-2.5-flash-native-audio-preview-09-2025" # Define it separately

CONFIG = types.LiveConnectConfig(
    response_modalities=["AUDIO"],
    output_audio_transcription={},
    # ... rest of config ...
    input_audio_transcription={},
    system_instruction={"parts": [{"text": SYSTEM_INSTRUCTION_TEXT}]},
    proactivity={"proactive_audio": True},
    speech_config=types.SpeechConfig(
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Zephyr")
        )
    ),
)
pya = pyaudio.PyAudio()

class AudioLoop:
    def __init__(self):
        self.session = None
        self.audio_in_queue = asyncio.Queue()
        self.input_stream = None
        self.output_stream = None
        
        # Initialize streams outside of run to allow cleanup in finally block
        try:
            self.output_stream = pya.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=24000, # Must match the model's output rate
                output=True,
                frames_per_buffer=CHUNK
            )
            # Explicitly using device 0, as confirmed by your diagnostic
            self.input_stream = pya.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK,
                input_device_index=0 
            )
        except Exception as e:
            print(f"Error opening audio streams: {e}")
            # Ensure cleanup if initialization fails
            if self.input_stream: self.input_stream.close()
            if self.output_stream: self.output_stream.close()
            pya.terminate()
            raise

    async def listen_audio(self):
        """Task to read audio from the microphone and send it to the session."""
        while True:
            # Use to_thread to make the blocking PyAudio read non-blocking for asyncio
            audio_data = await asyncio.to_thread(
                self.input_stream.read, CHUNK, exception_on_overflow=False
            )
            # Send audio data to the model
            await self.session.send(input={"data": audio_data, "mime_type": "audio/pcm"})

    async def receive_and_play_audio(self):
        """Task to receive responses from the session and handle output."""
        while True:
            # The session.receive() gives you an iterator over a single turn
            turn = self.session.receive()
            async for response in turn:
                # 1. Handle audio data
                if data := response.data:
                    self.audio_in_queue.put_nowait(data)
                    continue
                
                # 2. Handle text transcription
                # Note: The response.text field captures transcription text
                if text := response.text:
                    print("Model Transcript:", text, end="")

                # 3. Handle token usage and metadata
                if response.usage_metadata:
                    usage = response.usage_metadata
                    # Print usage when a turn completes or metadata is available
                    print(f"\n  Usage: Prompt Tokens: {usage.prompt_token_count}, Response Tokens: {usage.response_token_count}")

            # Crucial: Clear audio buffer if interrupted to enable low-latency interruptions
            while not self.audio_in_queue.empty():
                self.audio_in_queue.get_nowait()
    
    async def play_audio(self):
        """Task to consume audio from the queue and play it."""
        while True:
            bytestream = await self.audio_in_queue.get()
            # Use to_thread to make the blocking PyAudio write non-blocking for asyncio
            await asyncio.to_thread(self.output_stream.write, bytestream)
            self.audio_in_queue.task_done() # Tell the queue the item is processed


    async def run(self):
        """Main method to establish connection and run all tasks concurrently."""
        try:
            # Use the SDK's context manager for the connection
            async with (
                client.aio.live.connect(model=MODEL_NAME, config=CONFIG) as session,
                asyncio.TaskGroup() as tg,
            ):
                self.session = session
                print("Gemini Live Session Started. Press Ctrl+C to stop.")

                # Create concurrent tasks
                tg.create_task(self.listen_audio())
                tg.create_task(self.receive_and_play_audio())
                tg.create_task(self.play_audio())
                
                # We need a task that keeps the loop active, like asking for user text input
                # or just a long sleep that gets interrupted by Ctrl+C
                await asyncio.Future() 

        except asyncio.CancelledError:
            pass # Clean exit on Ctrl+C
        except Exception as e:
            print(f"An unexpected error occurred during execution: {e}")
        finally:
            # --- FINAL CLEANUP ---
            print("\nShutting down audio streams...")
            if self.input_stream:
                self.input_stream.stop_stream()
                self.input_stream.close()
            if self.output_stream:
                self.output_stream.stop_stream()
                self.output_stream.close()
            pya.terminate()
            print("Cleanup complete. Goodbye.")

if __name__ == "__main__":
    main = AudioLoop()
    # It's safer to wrap the run() call to capture KeyboardInterrupt
    try:
        asyncio.run(main.run())
    except KeyboardInterrupt:
        pass