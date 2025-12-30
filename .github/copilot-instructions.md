# Copilot Instructions for Gemini Live API Project

This document provides essential guidance for AI coding agents working on this codebase, focusing on the real-time audio interaction with the Google Gemini Live API.

## 1. Project Overview

This project facilitates real-time, bidirectional audio communication with the Gemini Live API. It captures microphone input, streams it to the Gemini model, and plays back the audio responses.

## 2. Core Components and Data Flow

The central logic resides within the `LiveApi.py` file, specifically the `AudioLoop` class.

-   **`AudioLoop` Class**: Manages the entire audio interaction lifecycle.
    -   `listen_audio()`: Captures audio from the local microphone.
    -   `send_realtime()`: Streams captured audio to the Gemini Live API.
    -   `receive_audio()`: Processes responses from the Gemini API, including audio data, model transcriptions, and usage metadata.
    -   `play_audio()`: Plays back the received audio responses.

**Data Flow**: Microphone -> `out_queue` -> Gemini API -> `audio_in_queue` -> Speaker.

## 3. Key Configuration and Dependencies

-   **Gemini API Key**: The `GEMINI_API_KEY` is loaded directly within `LiveApi.py` using `os.environ`. Ensure this environment variable is correctly set for local execution.
-   **System Instruction**: The model's behavior is primarily governed by the `system_instruction.txt` file. Modifications to this file will directly influence the Gemini model's responses and persona.
-   **Python Dependencies**:
    -   `pyaudio`: Essential for microphone input and speaker output.
    -   `google-generativeai`: The official client library for interacting with the Gemini API.
    -   Dependencies are listed in `requirements.txt`. To install them, run: `pip install -r requirements.txt`

## 4. Development Workflow

-   **Running the Application**: Execute `LiveApi.py` directly: `python LiveApi.py`
-   **Debugging**: The `receive_audio` method in `LiveApi.py` contains extensive `print` statements for debugging server content, model turns, input/output transcriptions, and token usage. This is the primary mechanism for understanding API interactions and model behavior.
-   **Asynchronous Operations**: The project heavily relies on `asyncio` for concurrent audio processing and API communication. Be mindful of `async`/`await` patterns when making changes.

## 5. Important Files

-   `LiveApi.py`: Contains the main application logic.
-   `system_instruction.txt`: Defines the Gemini model's system instruction.
-   `requirements.txt`: Lists Python dependencies.