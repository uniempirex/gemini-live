import asyncio
import sys
import traceback
import os  # Import the os module
import time

import pyaudio

from google import genai
# from google.generativeai import types # Import types for ModalityTokenCount

FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 256

pya = pyaudio.PyAudio()

# GEMINI_API_KEY must be set as env variable
print(f"GEMINI_API_KEY loaded: {bool(os.environ.get('GEMINI_API_KEY'))}")

client = genai.Client(http_options={"api_version": "v1alpha"})  # GEMINI_API_KEY must be set as env variable

# Load system instruction from file
with open("system_instruction.txt", "r") as f:
    system_instruction = f.read()

MODEL = "gemini-2.5-flash-native-audio-preview-12-2025"
CONFIG = {
    "system_instruction": system_instruction,
    "response_modalities": ["AUDIO"],
    "realtime_input_config": {
        "automatic_activity_detection": {
            "disabled": False, # default
            "start_of_speech_sensitivity": "START_SENSITIVITY_HIGH",
            "end_of_speech_sensitivity": "END_SENSITIVITY_HIGH",
            "prefix_padding_ms": 20,
            "silence_duration_ms": 10 }},
    "speech_config": {
        "voice_config": {"prebuilt_voice_config": {"voice_name": "erinome"}}},
    "enable_affective_dialog": False,
    "proactivity": {'proactive_audio': True},
    "output_audio_transcription": {},
    "input_audio_transcription": {},
}


class AudioLoop:
    def __init__(self):
        self.audio_in_queue = None
        self.out_queue = None
        self.session = None
        self.audio_stream = None
        self.output_audio_stream = None
        self.receive_audio_task = None
        self.play_audio_task = None
        self.initial_message_sent_time = None
        self.total_session_prompt_tokens = 0
        self.total_session_response_tokens = 0
        self.session_start_time = None
        self._server_content_printed = False # Initialize the flag

    async def listen_audio(self):
        mic_info = pya.get_default_input_device_info()
        self.audio_stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=SEND_SAMPLE_RATE,
            input=True,
            input_device_index=mic_info["index"],
            frames_per_buffer=CHUNK_SIZE,
        )
        if __debug__:
            kwargs = {"exception_on_overflow": False}
        else:
            kwargs = {}
        while True:
            data = await asyncio.to_thread(self.audio_stream.read, CHUNK_SIZE, **kwargs)
            await self.out_queue.put({"data": data, "mime_type": "audio/pcm"})

    async def send_realtime(self):
        while True:
            msg = await self.out_queue.get()
            await self.session.send_realtime_input(audio=msg)

    async def receive_audio(self):
        "Background task to reads from the websocket and write pcm chunks to the output queue"
        while True:
            turn = self.session.receive()
            async for response in turn:
                if data := response.data:
                    self.audio_in_queue.put_nowait(data)
                    continue
                
                if hasattr(response, 'server_content'):
                    # Print Server Content Details only if there's relevant content
                    if any(hasattr(response.server_content, attr) and getattr(response.server_content, attr) is not None
                           for attr in ['model_turn', 'output_transcription', 'input_transcription']):
                        if not self._server_content_printed: # Only print if not already printed for this turn
                            print("\n--- Server Content Details ---")
                            pydantic_internal_attrs = ['_abc_impl', 'model_computed_fields', 'model_config', 'model_extra', 'model_fields', 'model_fields_set']
                            for attr_name in dir(response.server_content):
                                if not attr_name.startswith('__') and attr_name not in pydantic_internal_attrs:
                                    attr_value = getattr(response.server_content, attr_name)
                                    if attr_name not in ['model_turn', 'output_transcription', 'input_transcription', 'generation_complete', 'interrupted'] and not callable(attr_value):
                                        print(f"  Server Content - {attr_name}: {attr_value}")
                            self._server_content_printed = True # Set the flag to True after printing

                    if hasattr(response.server_content, 'model_turn') and response.server_content.model_turn:
                        print("\n--- Model Turn Details ---")
                        for attr_name in dir(response.server_content.model_turn):
                            if not attr_name.startswith('__') and attr_name != 'parts' and attr_name not in pydantic_internal_attrs and not callable(getattr(response.server_content.model_turn, attr_name)):
                                print(f"  Model Turn - {attr_name}: {getattr(response.server_content.model_turn, attr_name)}")

                        # The model_turn contains the actual text response from the model
                        for i, part in enumerate(response.server_content.model_turn.parts):
                            print(f"\n--- Model Turn Part {i+1} Details ---")
                            for attr_name in dir(part):
                                if not attr_name.startswith('__') and attr_name not in pydantic_internal_attrs:
                                    attr_value = getattr(part, attr_name)
                                    if attr_name not in ['text', 'thought'] and not callable(attr_value):
                                        print(f"  Part {i+1} - {attr_name}: {attr_value}")
                                    elif attr_name == 'text' and attr_value:
                                        print("Output Transcript:", attr_value)
                                    elif attr_name == 'thought' and attr_value:
                                        print("Model Thought:", attr_value)

                    if hasattr(response.server_content, 'output_transcription') and response.server_content.output_transcription:
                        print("Output Transcription:", response.server_content.output_transcription.text)

                    if hasattr(response.server_content, 'input_transcription') and response.server_content.input_transcription:
                        print("Input Transcription:", response.server_content.input_transcription.text)

                    if hasattr(response.server_content, 'generation_complete') and response.server_content.generation_complete:
                        print("\nReceived: Generation Complete")

                    if hasattr(response.server_content, 'interrupted') and response.server_content.interrupted:
                        print("Received: Interrupted")

                # The server will periodically send messages that include UsageMetadata.
                if hasattr(response, 'usage_metadata') and (usage := response.usage_metadata):
                    # print("UsageMetadata object attributes:", dir(usage))
                    prompt_tokens = usage.prompt_token_count if usage.prompt_token_count is not None else 0
                    response_tokens = usage.response_token_count if usage.response_token_count is not None else 0 # Use response_token_count
                    total_tokens = usage.total_token_count if usage.total_token_count is not None else 0
                    print(f"  Usage: Prompt Tokens: {prompt_tokens}, Response Tokens: {response_tokens}, Total Tokens: {total_tokens}")
                    print(f"Used {total_tokens} tokens in total. Response token breakdown:")
                    if usage.response_tokens_details: # Added check for None
                        for detail in usage.response_tokens_details:
                            match detail:
                                case genai.types.ModalityTokenCount(modality=modality, token_count=count):
                                    print(f"{modality}: {count}")
                    
                    # Accumulate for session totals
                    self.total_session_prompt_tokens += prompt_tokens
                    self.total_session_response_tokens += response_tokens

            # After the turn is over
            print("\nReceived: Turn Complete")
            self._server_content_printed = False # Reset the flag for the next turn

            # If you interrupt the model, it sends a turn_complete.
            # For interruptions to work, we need to stop playback.
            # So empty out the audio queue because it may have loaded
            # much more audio than has played yet.
            while not self.audio_in_queue.empty():
                self.audio_in_queue.get_nowait()

    async def play_audio(self):
        self.output_audio_stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=RECEIVE_SAMPLE_RATE,
            output=True,
        )
        while True:
            bytestream = await self.audio_in_queue.get()
            await asyncio.to_thread(self.output_audio_stream.write, bytestream)

    async def run(self):
        self.session_start_time = time.time()
        print("--- Starting Gemini Live API Test ---")
        print("Press Ctrl+C to stop.")
        try:
            # Start the timer *before* the connect call, to include connection and setup time
            connect_start_time = time.time()
            async with (
                client.aio.live.connect(model=MODEL, config=CONFIG) as session,
                asyncio.TaskGroup() as tg,
            ):
                connect_end_time = time.time()
                initial_connect_latency = (connect_end_time - connect_start_time) * 1000
                print(f"Latency (connect call completion, including setup): {initial_connect_latency:.2f} ms")
                print("WebSocket Opened (and setup complete)")
                
                self.session = session

                self.audio_in_queue = asyncio.Queue()
                self.out_queue = asyncio.Queue(maxsize=5)

                tg.create_task(self.send_realtime())
                tg.create_task(self.listen_audio())
                tg.create_task(self.receive_audio())
                tg.create_task(self.play_audio())
        except asyncio.CancelledError:
            pass
        except ExceptionGroup as eg: # Changed asyncio.ExceptionGroup to ExceptionGroup
            if self.audio_stream:
                self.audio_stream.close()
            print(f"WebSocket Error: {eg}")
        except Exception as e:
            if self.audio_stream:
                self.audio_stream.close()
            print(f"WebSocket Error: {e}")
        finally:
            print("\n--- Stopping Gemini Live API Test ---")
            if self.audio_stream:
                self.audio_stream.stop_stream()
                self.audio_stream.close()
            if self.output_audio_stream:
                self.output_audio_stream.stop_stream()
                self.output_audio_stream.close()
            pya.terminate()
            print("Audio streams closed.")
            
            if self.session_start_time:
                session_end_time = time.time()
                session_duration = session_end_time - self.session_start_time
                print(f"\n--- Session Summary ---")
                print(f"Session Duration: {session_duration:.2f} seconds")
                print(f"Total Session Prompt Tokens: {self.total_session_prompt_tokens}")
                print(f"Total Session Response Tokens: {self.total_session_response_tokens}")
                print(f"Total Session Tokens (Prompt + Response): {self.total_session_prompt_tokens + self.total_session_response_tokens}")
            print("WebSocket Closed")


if __name__ == "__main__":
    loop = AudioLoop()
    asyncio.run(loop.run())