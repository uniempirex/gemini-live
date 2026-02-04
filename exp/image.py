import os
from google import genai
import PIL.Image

import os
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
organ = PIL.Image.open(os.path.join("media", '/Users/xiemprez/Pictures/screenshot/Screenshot 2025-10-27 at 22.38.35.png'))
response = client.models.generate_content_stream(
    model="gemini-2.0-flash", contents=["describe everything here, and breakdown what those explanations", organ]
)
for chunk in response:
    print(chunk.text)