! pip install -q google-genai gradio pillow

import os, io, base64, time
from io import BytesIO
from PIL import Image
import gradio as gr

from google import genai
from google.genai import types

# --- set your key here ---
os.environ["GEMINI_API_KEY"] = "YOUR API KEY"

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# models
TEXT_MODEL = "gemini-2.5-flash"  # fast, multi-modal chat
IMAGE_MODEL = "gemini-2.0-flash-preview-image-generation"  # native image generation/editing



def _extract_text(resp):
    """Return concatenated text from a generate_content response."""
    out = []
    try:
        # prefer parts for reliability
        for c in getattr(resp, "candidates", []) or []:
            for p in getattr(c, "content", {}).parts or []:
                if getattr(p, "text", None):
                    out.append(p.text)
        if out:
            return "\n".join(out).strip()
    except Exception:
        pass
    # fall back to resp.text if SDK provides it
    return getattr(resp, "text", "") or ""

def _extract_images(resp, save_prefix="gemini_image"):
    """
    Extract images from response parts (inline_data), return list of (PIL.Image, filename).
    Files are saved under /tmp (Colab) or current working dir (Jupyter) for convenience.
    """
    images = []
    idx = 0
    for c in getattr(resp, "candidates", []) or []:
        for p in getattr(c, "content", {}).parts or []:
            inline = getattr(p, "inline_data", None)
            if inline and getattr(inline, "data", None):
                img_bytes = inline.data
                # Some SDK variants already return bytes; others return base64.
                if isinstance(img_bytes, str):
                    img_bytes = base64.b64decode(img_bytes)
                img = Image.open(BytesIO(img_bytes)).convert("RGB")
                fname = f"{save_prefix}_{int(time.time())}_{idx}.png"
                img.save(fname)
                images.append((img, fname))
                idx += 1
    return images


def chat_reply(history, user_msg, system_prompt):
    """
    history: list of [user, assistant] for Gradio Chatbot
    user_msg: latest message
    system_prompt: optional system behavior text
    """
    if not user_msg.strip():
        return history, gr.update(value="")

    # build contents: we keep it simple → single turn message with optional system prefix
    preface = (system_prompt.strip() + "\n\n") if system_prompt else ""
    contents = preface + user_msg

    resp = client.models.generate_content(
        model=TEXT_MODEL,
        contents=contents
    )
    text = _extract_text(resp) or "(no response)"
    history = history + [[user_msg, text]]
    return history, gr.update(value="")

# ---- B) Image Generation ----
def generate_images(prompt):
    """
    Generate image(s) from text.
    Returns: gallery (list of PIL images), log text
    """
    if not prompt.strip():
        return [], "please enter a prompt."
    resp = client.models.generate_content(
        model=IMAGE_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=['TEXT', 'IMAGE']  # per docs: must include IMAGE
        )
    )
    # text (sometimes guidance/captions)
    caption = _extract_text(resp)
    imgs = _extract_images(resp, save_prefix="genimg")
    gallery = [img for img, _ in imgs]
    log = "generated images: " + (", ".join(f for _, f in imgs) if imgs else "none")
    if caption:
        log = (caption + "\n\n") + log
    return gallery, log

# ---- C) Image Editing ----
def edit_image(image, instruction):
    """
    Edit an uploaded image with a text instruction (e.g., 'turn the car blue').
    """
    if image is None:
        return None, "please upload an image."
    if not instruction.strip():
        return None, "please add an instruction."

    # Gradio gives us a PIL.Image
    pil_img = image.convert("RGBA") if image.mode != "RGBA" else image
    resp = client.models.generate_content(
        model=IMAGE_MODEL,
        contents=[instruction, pil_img],
        config=types.GenerateContentConfig(
            response_modalities=['TEXT', 'IMAGE']
        )
    )
    caption = _extract_text(resp)
    imgs = _extract_images(resp, save_prefix="edited")
    out = imgs[0][0] if imgs else None
    log = caption or ""
    if imgs:
        log += ("\n\nsaved: " + imgs[0][1])
    return out, log.strip() or "done."

# ---- D) Vision Q&A / Image Understanding ----
def analyze_image(image, question):
    """
    Ask Gemini about the uploaded image (describe, classify, extract info, etc.).
    """
    if image is None:
        return "please upload an image."
    prompt = question.strip() or "Describe this image in detail."
    resp = client.models.generate_content(
        model=TEXT_MODEL,
        contents=[prompt, image]  # multimodal input
    )
    return _extract_text(resp) or "(no answer)"


with gr.Blocks(title="Gemini All-in-One Chatbot") as demo:
    gr.Markdown("# ✨ Gemini All-in-One\nText chat • Image generation • Image editing • Vision Q&A")

    with gr.Tab("💬 Chat"):
        with gr.Row():
            system_box = gr.Textbox(label="system prompt (optional)", placeholder="e.g., You are a helpful coding assistant.")
        chatbot = gr.Chatbot(height=420, type="messages")
        with gr.Row():
            msg = gr.Textbox(show_label=False, placeholder="type your message and press enter…")
        clear = gr.Button("clear chat")

        chat_state = gr.State([])  # [[user, assistant], ...]

        def _on_send(user_text, hist, sys):
            return chat_reply(hist, user_text, sys)

        msg.submit(_on_send, [chat_state, msg, system_box], [chat_state, msg])
        clear.click(lambda: ([], ""), None, [chat_state, msg])

        # keep the Chatbot widget in sync
        def _render_history(hist):
            return hist
        chat_state.change(_render_history, chat_state, chatbot)

    with gr.Tab("🖼️ Image Generator"):
        prompt = gr.Textbox(label="prompt", placeholder="a cozy cabin under the northern lights, ultra wide, cinematic")
        go = gr.Button("generate")
        gallery = gr.Gallery(label="results", height=420, columns=2)
        gen_log = gr.Markdown()

        go.click(generate_images, prompt, [gallery, gen_log])

    with gr.Tab("🪄 Image Editor"):
        with gr.Row():
            img_in = gr.Image(label="upload image", type="pil")
            instruction = gr.Textbox(label="edit instruction", placeholder="add a soft golden-hour glow")
        edit_btn = gr.Button("edit")
        img_out = gr.Image(label="edited image")
        edit_log = gr.Markdown()
        edit_btn.click(edit_image, [img_in, instruction], [img_out, edit_log])

    with gr.Tab("🔍 Vision Q&A"):
        with gr.Row():
            vis_img = gr.Image(label="upload image", type="pil")
            vis_q = gr.Textbox(label="ask about the image", placeholder="what's happening here?")
        ask_btn = gr.Button("ask")
        vis_ans = gr.Markdown()
        ask_btn.click(analyze_image, [vis_img, vis_q], vis_ans)

demo.launch()
