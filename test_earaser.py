import time
from PIL import Image
from simple_lama_inpainting import SimpleLama

# 1. Initialize the Engine
print("🚀 Loading LaMa Engine...")
start_load = time.time()
simple_lama = SimpleLama()
print(f"✅ Model loaded in {time.time() - start_load:.2f}s")

def run_test(img_path, mask_path, output_path):
    # Load your files
    img = Image.open(img_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    
    print(f"🪄 Starting inpainting on {img.size} image...")
    start_time = time.time()
    
    # Run the FFC model
    result = simple_lama(img, mask)
    
    duration = time.time() - start_time
    print(f"✅ Finished in {duration:.2f}s")
    result.save(output_path)
    print(f"💾 Result saved to {output_path}")

if __name__ == "__main__":
    # Ensure you have 'test_image.png' and 'test_mask.png' in your folder
    try:
        run_test("test_image.png", "test_mask.png", "test_result.png")
    except FileNotFoundError:
        print("❌ Error: Place a 'test_image.png' and 'test_mask.png' in this folder first!")