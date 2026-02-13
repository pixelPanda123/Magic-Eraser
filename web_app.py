import gradio as gr
import numpy as np
import torch
from PIL import Image
from simple_lama_inpainting import SimpleLama

# 1. Device check - M-series Macs use 'mps'
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"🚀 Magic Eraser running on: {device}")

# 2. Global Initialization
print("Loading LaMa weights...")
simple_lama = SimpleLama() 

def process_image(input_img):
    if input_img is None:
        return None

    # Get the original image (Linus wallpaper)
    bg = input_img["background"].convert("RGB")
    
    # 3. Handle Resolution
    # Your uploaded result was 1920x1080, but the original was 1024x576.
    # We resize to a standard 1024px to keep it snappy on the Mac.
    max_dim = 1024
    original_size = bg.size
    if max(bg.size) > max_dim:
        bg.thumbnail((max_dim, max_dim), Image.LANCZOS)
    
    # 4. Extract Mask from Layers
    layers = input_img.get("layers", [])
    if not layers:
        return bg

    # Use Alpha channel to find where you brushed
    mask_layer = layers[0].convert("RGBA")
    alpha = mask_layer.getchannel('A')
    mask_np = np.array(alpha)
    
    # Convert to binary (White = erase)
    mask_np = np.where(mask_np > 0, 255, 0).astype(np.uint8)
    final_mask = Image.fromarray(mask_np).convert("L")
    
    # Match mask size to background
    final_mask = final_mask.resize(bg.size, resample=Image.NEAREST)

    print(f"Inpainting {bg.size} image...")
    
    # This matches the global variable name now
    result = simple_lama(bg, final_mask)
    
    # Return to original resolution for the user
    return result.resize(original_size, Image.LANCZOS)

# UI Layout
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("Bepop eraser")
    gr.Markdown("Brush over any object to remove them using Fast Fourier Convolutions.")
    
    with gr.Row():
        img_input = gr.ImageEditor(
            label="Input", 
            type="pil", 
            eraser=True,
            brush=gr.Brush(colors=["#000000"]) # Contrast color for the UI
        )
        img_output = gr.Image(label="Result")
    
    with gr.Row():
        submit_btn = gr.Button("Remove Object", variant="primary")
        clear_btn = gr.Button("Clear")

    submit_btn.click(process_image, inputs=[img_input], outputs=[img_output])
    clear_btn.click(lambda: (None, None), outputs=[img_input, img_output])

if __name__ == "__main__":
    demo.launch()