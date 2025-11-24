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
                           rate=RATE,
                           output=True,
                           frames_per_buffer=CHUNK)

# WebSocket setup
ws = None

def on_message(ws, message):
    try:
        response_data = json.loads(message)
        if "serverContent" in response_data and "modelTurn" in response_data["serverContent"] and "parts" in response_data["serverContent"]["modelTurn"]:
            for part in response_data["serverContent"]["modelTurn"]["parts"]:
                if "inlineData" in part and "data" in part["inlineData"]:
                    returned_audio_data = base64.b64decode(part["inlineData"]["data"])
                    if returned_audio_data:
                        output_stream.write(returned_audio_data)
    except json.JSONDecodeError:
        print(f"Received non-JSON message: {message}")
    except Exception as e:
        print(f"Error processing WebSocket message: {e}")

def on_error(ws, error):
    print(f"WebSocket Error: {error}")

def on_close(ws, close_status_code, close_msg):
    print("WebSocket Closed")

def on_open(ws):
    print("WebSocket Opened")
    # Send initial session configuration
    setup_message = {
        "setup": {
            "model": "models/gemini-2.5-flash-native-audio-latest", # Specify the model
            "generationConfig": {
                "responseModalities": ["AUDIO"], # Request audio responses
            },
            # Removed speechConfig as it's not a recognized field in the raw WebSocket setup message
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
