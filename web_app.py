import gradio as gr
import numpy as np
import torch
from PIL import Image
from simple_lama_inpainting import SimpleLama

# 1. Device check - Optimizing for your Mac's M-series GPU
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"NovaVision 01 running on: {device}")

# 2. Global Initialization
print("Loading LaMa weights...")
simple_lama = SimpleLama() 

# --- CORE FUNCTIONS ---

def process_image(input_img):
    """
    Core inpainting logic using Fast Fourier Convolutions (FFC).
    """
    if input_img is None:
        return None
        
    # Extract background (original photo)
    bg = input_img["background"].convert("RGB")
    
    # 3. Handle Resolution for M-series performance
    max_dim = 1024
    original_size = bg.size
    if max(bg.size) > max_dim:
        bg.thumbnail((max_dim, max_dim), Image.LANCZOS)
    
    # 4. Extract Mask from Layers
    layers = input_img.get("layers", [])
    if not layers:
        return bg

    # Convert brush strokes to a binary mask (White = erase)
    mask_layer = layers[0].convert("RGBA")
    alpha = mask_layer.getchannel('A')
    mask_np = np.array(alpha)
    mask_np = np.where(mask_np > 0, 255, 0).astype(np.uint8)
    
    final_mask = Image.fromarray(mask_np).convert("L")
    
    # Ensure mask dimensions match the background
    final_mask = final_mask.resize(bg.size, resample=Image.NEAREST)

    print(f"Inpainting {bg.size} image...")
    result = simple_lama(bg, final_mask)
    
    # Return to original resolution for the final output
    return result.resize(original_size, Image.LANCZOS)

def process_and_log(input_img, history):
    """
    Handles both the removal process and updating the visual session log.
    """
    if input_img is None:
        return None, history
        
    result = process_image(input_img)
    
    if history is None:
        history = []
    history.insert(0, result) # Add new result to the top of the gallery
    
    return result, history

# --- THEME & STYLING ---

# Defining the Dual-Cream Palette
# Page Background (Darker): #F3E5AB (Vanilla)
# Container Background (Lighter): #FFFDD0 (Cream)

cream_theme = gr.themes.Default(
    primary_hue="indigo",
    secondary_hue="slate",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Plus Jakarta Sans"), "ui-sans-serif", "sans-serif"],
).set(
    # Page-wide background
    body_background_fill="#F3E5AB", 
    
    # Component-level background
    block_background_fill="#FFFDD0",
    block_border_width="3px",
    block_border_color="#000000", 
    
    # High-Contrast Black Text
    body_text_color="#000000",
    block_title_text_color="#000000",
    block_label_text_color="#000000",
    block_title_text_weight="900",
    
    # Navigation & Interaction
    button_primary_background_fill="#000000",
    button_primary_text_color="#FFFDD0",
    button_secondary_background_fill="transparent",
    button_secondary_border_color="#000000",
    button_secondary_text_color="#000000"
)

# Custom CSS for the full-page cream coverage and typography pop
custom_css = """
body, .gradio-container { 
    background-color: #F3E5AB !important; 
}

.container { 
    max-width: 850px; 
    margin: 40px auto; 
    padding: 40px; 
    background-color: #FFFDD0; 
    border: 4px solid #000000;
}

h1 { 
    font-size: 3.5rem; 
    font-weight: 900; 
    color: #000000; 
    letter-spacing: -0.05em; 
    margin-bottom: 0.1rem; 
}

.description { 
    font-size: 1.1rem; 
    color: #000000; 
    margin-bottom: 2rem; 
    font-weight: 800; 
    text-transform: uppercase; 
    border-bottom: 2px solid #000000; 
    display: inline-block; 
}

.history-title { 
    margin-top: 4rem; 
    font-weight: 900; 
    font-size: 2rem; 
    color: #000000; 
    border-top: 4px solid #000000; 
    padding-top: 1rem; 
}
"""

# --- UI LAYOUT ---

with gr.Blocks() as demo:
    with gr.Column(elem_classes="container"):
        gr.HTML("<h1>Object eraser</h1>")
        gr.HTML("<p class='description'>Fourier-Domain Generative Inpainting</p>")

        with gr.Column():
            img_input = gr.ImageEditor(
                label="SOURCE CANVAS",
                type="pil",
                interactive=True,
                # Black brush for aesthetic consistency with the Noir theme
                brush=gr.Brush(colors=["#000000"], color_mode="fixed")
            )
        
        with gr.Row():
            run_btn = gr.Button("EXECUTE REMOVAL", variant="primary", size="lg")
            clear_btn = gr.Button("RESET SESSION", variant="secondary", size="lg")

        with gr.Column():
            img_output = gr.Image(label="GENERATED OUTPUT", interactive=False)

        gr.HTML("<div class='history-title'>LOGS / HISTORY</div>")
        history_display = gr.Gallery(
            columns=4, 
            object_fit="cover", 
            height="auto",
            show_label=False
        )
        
        # Internal state to store the history of images
        history_state = gr.State([])

    # Event Handlers
    run_btn.click(
        fn=process_and_log,
        inputs=[img_input, history_state],
        outputs=[img_output, history_display]
    )
    
    clear_btn.click(
        lambda: (None, None, []), 
        outputs=[img_input, img_output, history_display]
    )

if __name__ == "__main__":
    # Launch with the theme and CSS passed directly (Gradio 6.0 Standard)
    demo.launch(theme=cream_theme, css=custom_css)