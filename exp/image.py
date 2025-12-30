import os
from google import genai
import PIL.Image

client = genai.Client(api_key="AIzaSyCdlmDj1rJhq91RQwiw5F3rFyCFKzmbGmk")
organ = PIL.Image.open(os.path.join("media", '/Users/xiemprez/Pictures/screenshot/Screenshot 2025-10-27 at 22.38.35.png'))
response = client.models.generate_content_stream(
    model="gemini-2.0-flash", contents=["describe everything here, and breakdown what those explanations", organ]
)
for chunk in response:
    print(chunk.text)