import pyaudio
import websocket
import threading
import json
import base64
import time

# --- Configuration ---
FORMAT = pyaudio.paInt16  # Audio format
CHANNELS = 1              # Mono audio
RATE = 16000              # Sample rate (Hz)
CHUNK = 1024              # Audio chunk size

API_KEY = "AIzaSyCdlmDj1rJhq91RQwiw5F3rFyCFKzmbGmk" # Your Gemini API key
GEMINI_LIVE_API_URL = f"wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent?key={API_KEY}" # Official Gemini Live API WebSocket endpoint with API key as query parameter

# --- PyAudio Setup ---
audio = pyaudio.PyAudio()

# Open input stream (microphone)
input_stream = audio.open(format=FORMAT,
                          channels=CHANNELS,
                          rate=RATE,
                          input=True,
                          frames_per_buffer=CHUNK)

# Open output stream (speakers)
output_stream = audio.open(format=FORMAT,
                           channels=CHANNELS,
                           rate=24000,
                           output=True,
                           frames_per_buffer=CHUNK)

# Variable to store the start time for initial response latency calculation
initial_message_sent_time = None

# Variables to store cumulative token counts for the entire session
total_session_prompt_tokens = 0
total_session_response_tokens = 0

# Variable to store the start time of the entire session
session_start_time = time.time()

# WebSocket setup
ws = None

def on_message(ws, message):
    global initial_message_sent_time
    global total_session_prompt_tokens
    global total_session_response_tokens
    try:
        response_data = json.loads(message)

        # Calculate latency for the very first valid JSON response received
        # after the initial setup message was sent.
        if initial_message_sent_time is not None:
            latency = (time.time() - initial_message_sent_time) * 1000 # Latency in milliseconds
            print(f"Latency (initial setup message sent to first server response): {latency:.2f} ms")
            initial_message_sent_time = None # Reset after the first response is processed

        # Now process the content of the message
        if "serverContent" in response_data and "modelTurn" in response_data["serverContent"] and "parts" in response_data["serverContent"]["modelTurn"]:
            for part in response_data["serverContent"]["modelTurn"]["parts"]:
                if "inlineData" in part and "data" in part["inlineData"]:
                    returned_audio_data = base64.b64decode(part["inlineData"]["data"])
                    if returned_audio_data:
                        output_stream.write(returned_audio_data)
        elif "setupResponse" in response_data:
            print(f"Received setup response: {response_data['setupResponse']}")
        elif "setupComplete" in response_data:
            print("Received: Setup Complete")
        elif "serverContent" in response_data:
            server_content = response_data["serverContent"]
            if "generationComplete" in server_content and server_content["generationComplete"]:
                print("Received: Generation Complete")
            elif "turnComplete" in server_content and server_content["turnComplete"]:
                print("Received: Turn Complete")
                if "usageMetadata" in response_data:
                    usage = response_data["usageMetadata"]
                    prompt_tokens = usage.get('promptTokenCount', 0)
                    response_tokens = usage.get('responseTokenCount', 0)
                    total_tokens = usage.get('totalTokenCount', 0)
                    print(f"  Usage: Prompt Tokens: {prompt_tokens}, Response Tokens: {response_tokens}, Total Tokens: {total_tokens}")
                    
                    # Accumulate for session totals
                    total_session_prompt_tokens += prompt_tokens
                    total_session_response_tokens += response_tokens
            elif "interrupted" in server_content and server_content["interrupted"]:
                print("Received: Interrupted")
            else:
                print(f"Received unhandled serverContent: {server_content}")
        else:
            print(f"Received unhandled JSON message: {response_data}")

    except json.JSONDecodeError:
        print(f"Received non-JSON message: {message}")
    except Exception as e:
        print(f"Error processing WebSocket message: {e}")

def on_error(ws, error):
    print(f"WebSocket Error: {error}")

def on_close(ws, close_status_code, close_msg):
    print("WebSocket Closed")

def on_open(ws):
    global initial_message_sent_time
    print("WebSocket Opened")
    initial_message_sent_time = time.time() # Record time when WebSocket is opened (includes sending setup message)
    # Send initial session configuration
    setup_message = {
        "setup": {
            "model": "models/gemini-2.5-flash-native-audio-preview-09-2025", # Specify the model
            "generationConfig": {
                "responseModalities": ["AUDIO"], # Request audio responses
            },
            # Add system instruction here
            "systemInstruction": {
                "parts": [
                    {"text": "You are a mighty legendary philosopher. Use only Indonesian language."}
                ]
            }
        }
    }
    ws.send(json.dumps(setup_message))

print("--- Starting Gemini Live API Test ---")
print("Press Ctrl+C to stop.")

# Start WebSocket in a separate thread
ws = websocket.WebSocketApp(GEMINI_LIVE_API_URL,
                            on_open=on_open,
                            on_message=on_message,
                            on_error=on_error,
                            on_close=on_close)

ws_thread = threading.Thread(target=ws.run_forever)
ws_thread.daemon = True
ws_thread.start()

try:
    while True:
        # 1. Capture audio from microphone
        audio_data = input_stream.read(CHUNK, exception_on_overflow=False)
        
        # 2. Send audio to Gemini Live API via WebSocket
        if ws and ws.sock and ws.sock.connected:
            realtime_input_message = {
                "realtimeInput": {
                    "audio": {
                        "data": base64.b64encode(audio_data).decode('utf-8'),
                        "mimeType": "audio/pcm"
                    }
                }
            }
            ws.send(json.dumps(realtime_input_message), websocket.ABNF.OPCODE_TEXT) # Send as text with JSON
        
        # Small delay to prevent excessive CPU usage, adjust as needed
        time.sleep(0.01)

except KeyboardInterrupt:
    print("--- Stopping Gemini Live API Test ---")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
finally:
    # --- Cleanup ---
    if ws:
        ws.close()
    input_stream.stop_stream()
    input_stream.close()
    output_stream.stop_stream()
    output_stream.close()
    audio.terminate()
    print("Audio streams closed.")
    
    # Calculate and display session duration
    session_end_time = time.time()
    session_duration = session_end_time - session_start_time
    print(f"\n--- Session Summary ---")
    print(f"Session Duration: {session_duration:.2f} seconds")
    
    # Display total session token usage
    print(f"Total Session Prompt Tokens: {total_session_prompt_tokens}")
    print(f"Total Session Response Tokens: {total_session_response_tokens}")
    print(f"Total Session Tokens (Prompt + Response): {total_session_prompt_tokens + total_session_response_tokens}")
