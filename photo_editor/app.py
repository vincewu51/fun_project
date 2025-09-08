import gradio as gr
from openai import OpenAI
from io import BytesIO
from PIL import Image
import time

client = OpenAI(api_key="GEMINI_API_KEY")

# Predefined fixed image (change path to your fixed person photo)
FIXED_IMAGE_PATH = "fixed_person.jpg"
fixed_img = Image.open(FIXED_IMAGE_PATH)

def edit_images(img2):
    # Convert predefined img1
    img_bytes1 = BytesIO()
    fixed_img.save(img_bytes1, format="JPEG")
    img_data1 = img_bytes1.getvalue()

    # Convert user-uploaded img2
    img_bytes2 = BytesIO()
    img2.save(img_bytes2, format="JPEG")
    img_data2 = img_bytes2.getvalue()

    prompt = (
        "Take the people from the first photo and seamlessly place them into the background "
        "of the second photo. Remove the people and other objects from the second photo. "
        "Preserve all facial features, body proportions, and clothing details exactly. "
        "Convert the style of the first photo into the style of the second photo. "
        "Maintain photorealistic colors, textures, and details for both the people and the background."
    )

    response = client.models.generate_content(
        model="gemini-2.5-flash-image-preview",
        contents=[
            {"inline_data": {"mime_type": "image/jpeg", "data": img_data1}},
            {"inline_data": {"mime_type": "image/jpeg", "data": img_data2}},
            prompt,
        ],
    )

    if response.candidates:
        for part in response.candidates[0].content.parts:
            if hasattr(part, "inline_data") and part.inline_data is not None:
                img_bytes = part.inline_data.data
                image = Image.open(BytesIO(img_bytes))

                # Unique filename
                timestamp = int(time.time() * 1000)
                filename = f"edited_{timestamp}.png"
                image.save(filename)

                return image
    return None


demo = gr.Interface(
    fn=edit_images,
    inputs=gr.Image(type="pil", label="Upload Background Photo"),
    outputs=gr.Image(type="pil", label="Result"),
    title="Photo Editor - People into Background",
    description="Upload one background image. The app will insert the fixed people photo into your background."
)

if __name__ == "__main__":
    demo.launch()
