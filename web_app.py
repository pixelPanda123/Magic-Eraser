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

# Clean White Minimal Theme
cream_theme = gr.themes.Default(
    primary_hue="indigo",
    secondary_hue="slate",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Plus Jakarta Sans"), "ui-sans-serif", "sans-serif"],
).set(
    # Page-wide background
    body_background_fill="#FFFFFF", 
    
    # Component-level background
    block_background_fill="#FFFFFF",
    block_border_width="1px",
    block_border_color="#E5E7EB",  # light grey
    
    # Text
    body_text_color="#0F172A",        # near-black
    block_title_text_color="#0F172A",
    block_label_text_color="#334155", # slate grey
    block_title_text_weight="700",
    
    # Buttons
    button_primary_background_fill="#1E293B",  # dark slate
    button_primary_text_color="#FFFFFF",
    button_secondary_background_fill="transparent",
    button_secondary_border_color="#CBD5E1",
    button_secondary_text_color="#1E293B"
)

# Clean white layout + modern spacing
custom_css = """
body, .gradio-container { 
    background-color: #F8FAFC !important;  /* soft off-white */
}

.container { 
    max-width: 900px; 
    margin: 48px auto; 
    padding: 48px; 
    background-color: #FFFFFF; 
    border-radius: 14px;
    border: 1px solid #E5E7EB;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.06);
}

h1 { 
    font-size: 3rem; 
    font-weight: 800; 
    color: #020617; 
    letter-spacing: -0.03em; 
    margin-bottom: 0.3rem; 
}

.description { 
    font-size: 1rem; 
    color: #475569; 
    margin-bottom: 2rem; 
    font-weight: 600; 
    text-transform: none; 
    border-bottom: 2px solid #E5E7EB; 
    display: inline-block; 
    padding-bottom: 0.4rem;
}

.history-title { 
    margin-top: 4rem; 
    font-weight: 700; 
    font-size: 1.8rem; 
    color: #020617; 
    border-top: 1px solid #E5E7EB; 
    padding-top: 1.2rem; 
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