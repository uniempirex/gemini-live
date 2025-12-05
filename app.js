const startButton = document.getElementById('startButton');
const stopButton = document.getElementById('stopButton');
const apiKeyInput = document.getElementById('apiKey');
const statusDiv = document.getElementById('status');

let audioContext;
let mediaStream;
let scriptProcessor;
let websocket;
let audioQueue = [];
let isPlaying = false;

const CHUNK_SIZE = 512;
const SAMPLE_RATE = 16000;
const OUTPUT_SAMPLE_RATE = 24000; // Gemini Live output sample rate

startButton.addEventListener('click', () => {
    const apiKey = apiKeyInput.value.trim();
    if (!apiKey) {
        alert('Please enter your Gemini API Key.');
        return;
    }
    startSession(apiKey);
});

stopButton.addEventListener('click', () => {
    stopSession();
});

async function startSession(apiKey) {
    statusDiv.textContent = 'Starting...';
    startButton.disabled = true;
    stopButton.disabled = false;

    // Fetch system instruction content
    let systemInstructionContent = "";
    try {
        const response = await fetch('/system_instruction.txt');
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        systemInstructionContent = await response.text();
        statusDiv.textContent = 'System instruction loaded. Connecting to Gemini Live API...';
    } catch (error) {
        console.error('Error loading system instruction:', error);
        statusDiv.textContent = 'Error loading system instruction. Please check console.';
        startButton.disabled = false;
        stopButton.disabled = true;
        return; // Stop session if instruction fails to load
    }

    audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: SAMPLE_RATE });

    const GEMINI_LIVE_API_URL = `wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1alpha.GenerativeService.BidiGenerateContent?key=${apiKey}`;

    websocket = new WebSocket(GEMINI_LIVE_API_URL);

    websocket.onopen = () => {
        statusDiv.textContent = 'WebSocket opened. Sending setup message...';
        const setupMessage = {
            "setup": {
                "model": "models/gemini-2.5-flash-native-audio-preview-09-2025",
                "generationConfig": {
                    "responseModalities": ["AUDIO"],
                },
                "outputAudioTranscription": {},
                "inputAudioTranscription": {},
                "systemInstruction": {
                    "parts": [
                        {"text": systemInstructionContent}
                    ]
                },
                "proactivity": {
                    "proactiveAudio": true
                }
            }
        };
        websocket.send(JSON.stringify(setupMessage));
        startMicrophone();
    };

    websocket.onmessage = (event) => {
        if (event.data instanceof Blob) {
            const reader = new FileReader();
            reader.onload = () => {
                try {
                    processWebSocketMessage(reader.result);
                } catch (e) {
                    console.error("Error processing Blob as JSON:", e);
                }
            };
            reader.readAsText(event.data);
        } else {
            processWebSocketMessage(event.data);
        }
    };

    function processWebSocketMessage(message) {
        const responseData = JSON.parse(message);
        if (responseData.serverContent && responseData.serverContent.modelTurn && responseData.serverContent.modelTurn.parts) {
            for (const part of responseData.serverContent.modelTurn.parts) {
                if (part.inlineData && part.inlineData.data) {
                    const audioData = atob(part.inlineData.data);
                    audioQueue.push(audioData);
                    if (!isPlaying) {
                        playNextAudioChunk();
                    }
                }
            }
        } else if (responseData.setupResponse) {
            statusDiv.textContent = `Received setup response: ${JSON.stringify(responseData.setupResponse)}`;
        } else if (responseData.setupComplete) {
            statusDiv.textContent = 'Received: Setup Complete';
        } else if (responseData.serverContent) {
            const serverContent = responseData.serverContent;
            if (serverContent.generationComplete) {
                console.log("Received: Generation Complete");
            } else if (serverContent.turnComplete) {
                console.log("Received: Turn Complete");
                if (responseData.usageMetadata) {
                    const usage = responseData.usageMetadata;
                    console.log(`  Usage: Prompt Tokens: ${usage.promptTokenCount || 0}, Response Tokens: ${usage.responseTokenCount || 0}, Total Tokens: ${usage.totalTokenCount || 0}`);
                }
            } else if (serverContent.interrupted) {
                console.log("Received: Interrupted");
            } else if (serverContent.outputTranscription && serverContent.outputTranscription.text) {
                console.log("Output Transcript:", serverContent.outputTranscription.text);
            } else if (serverContent.outputTranscription) {
                // Only log if it's not an empty object, which is handled by the above condition
                console.log(`Received unhandled outputTranscription: ${JSON.stringify(serverContent.outputTranscription)}`);
            } else if (serverContent.inputTranscription && serverContent.inputTranscription.text) {
                console.log("Input Transcript:", serverContent.inputTranscription.text);
            } else {
                console.log(`Received unhandled serverContent: ${JSON.stringify(serverContent)}`);
            }
        } else {
            console.log(`Received unhandled JSON message: ${JSON.stringify(responseData)}`);
        }
    }
}

function stopSession() {
    if (mediaStream) {
        mediaStream.getTracks().forEach(track => track.stop());
    }
    if (scriptProcessor) {
        scriptProcessor.disconnect();
    }
    if (audioContext) {
        audioContext.close();
        audioContext = null; // Clear the context
    }
    if (websocket) {
        websocket.close();
    }
    audioQueue = [];
    isPlaying = false;
    startButton.disabled = false;
    stopButton.disabled = true;
    statusDiv.textContent = 'Session stopped.';
}

async function startMicrophone() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            audio: {
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true, // Added autoGainControl
                sampleRate: SAMPLE_RATE
            }
        });
        mediaStream = stream;
        const source = audioContext.createMediaStreamSource(stream);
        scriptProcessor = audioContext.createScriptProcessor(CHUNK_SIZE, 1, 1);

        scriptProcessor.onaudioprocess = (e) => {
            const inputData = e.inputBuffer.getChannelData(0);
            const audioData = convertFloat32ToInt16(inputData);
            if (websocket && websocket.readyState === WebSocket.OPEN) {
                const realtimeInputMessage = {
                    "realtimeInput": {
                        "audio": {
                            "data": btoa(String.fromCharCode.apply(null, new Uint8Array(audioData.buffer))),
                            "mimeType": "audio/pcm"
                        }
                    }
                };
                websocket.send(JSON.stringify(realtimeInputMessage));
            }
        };

        source.connect(scriptProcessor);
        scriptProcessor.connect(audioContext.destination);
        statusDiv.textContent = 'Microphone started. You can start speaking.';
    } catch (error) {
        console.error('Error accessing microphone:', error);
        statusDiv.textContent = 'Error accessing microphone. See console for details.';
        stopSession();
    }
}

async function playNextAudioChunk() {
    if (audioQueue.length === 0 || isPlaying) {
        return;
    }

    isPlaying = true;
    const audioData = audioQueue.shift();

    // Gemini Live API returns raw PCM (signed 16-bit little-endian) at 24000 Hz
    const numChannels = 1; // Mono audio
    const bytesPerSample = 2; // Int16

    // Create an ArrayBuffer from the base64 decoded string
    const pcmData = new Int16Array(audioData.length / bytesPerSample);
    for (let i = 0; i < audioData.length / bytesPerSample; i++) {
        // Little-endian: byte 0 is LSB, byte 1 is MSB
        const byte1 = audioData.charCodeAt(i * bytesPerSample);
        const byte2 = audioData.charCodeAt(i * bytesPerSample + 1);
        pcmData[i] = byte1 | (byte2 << 8);
    }

    // Use the existing audioContext for playback
    // We need to ensure the audioContext's sample rate is 24000 for playback
    // If the input sample rate is different, we might need resampling, but for now,
    // we'll assume the output is always 24000 Hz.
    const audioBuffer = audioContext.createBuffer(numChannels, pcmData.length, OUTPUT_SAMPLE_RATE);
    const nowBuffering = audioBuffer.getChannelData(0);
    for (let i = 0; i < pcmData.length; i++) {
        nowBuffering[i] = pcmData[i] / 32768; // Normalize to float32
    }

    const source = audioContext.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(audioContext.destination);

    source.onended = () => {
        isPlaying = false;
        playNextAudioChunk(); // Play the next chunk in the queue
    };

    source.start();
}

function convertFloat32ToInt16(buffer) {
    let l = buffer.length;
    const buf = new Int16Array(l);
    while (l--) {
        buf[l] = Math.min(1, buffer[l]) * 0x7FFF;
    }
    return buf;
}
