# NovaVision: The Magic Eraser

> "David Copperfield is a famous fictional character and also a magician, who is particularly known for his disappearing acts. That's what I intended with this project: to disappear objects that are not desired by the user."

This project is a high-performance image inpainting pipeline designed to seamlessly remove objects from any scene. It utilizes state-of-the-art deep learning architectures to ensure that the "disappearance" is not just a blur, but a mathematically consistent reconstruction of the background.

---

## Methodology

To achieve professional-grade results, I employed a hybrid pipeline consisting of two essential techniques: **Large Mask Inpainting (LaMa)** and the **Segment Anything Model (SAM)**.

### 1. LaMa: Large Mask Inpainting

Large Mask Inpainting is based on a novel network architecture that uses **Fast Fourier Convolutions (FFCs)** and a high receptive field perceptual loss. While standard CNNs struggle with large holes due to limited receptive fields, LaMa can synthesize complex textures and structures by operating in the frequency domain.

**The FFC Operator:**
Fast Fourier Convolution (FFC) allows the model to utilize global context even in the earliest layers of the network. The process begins by taking a color image  masked by a binary mask  of unknown pixels. This masked image is stacked with the mask  to create a four-channel input, which is then fed into the generator.

The spectral branch of the FFC performs the following operation:

1. **Transform**: Applies a Discrete Fourier Transform (DFT) to the input features.
2. **Global Interaction**: Learns weights in the frequency domain, where a single point represents global information.
3. **Inverse**: Applies an Inverse DFT (IDFT) to map the features back to the spatial domain.

This ensures that if you remove a person from a brick wall, the model "sees" the pattern of bricks across the entire image and reconstructs the missing area with identical geometry.

### 2. Segment Anything (SAM)

While LaMa handles the "disappearing act," **Segment Anything** acts as the high-precision "assistant" that identifies the object to be removed.

**Image Encoder & Mask Decoder:**
SAM uses a heavy **Vision Transformer (ViT)** backbone to extract image features. These features are pre-computed (encoded) only once per image to ensure real-time responsiveness.

* **Promptable Interface**: The user provides a "prompt" (a brush stroke or a click).
* **Cross-Attention**: The lightweight mask decoder uses bidirectional cross-attention between the user's prompt and the encoded image features.
* **Zero-Shot Generalization**: Because it is a foundation model, SAM can accurately find the boundaries of any object—from a person in a wallpaper to a specific leaf in a tomato plant—without needing specific training on that object.

---

## Design Aesthetic: Dual-Chroma Minimalist

The user interface is designed with a high-contrast **Dual-Cream & Noir** aesthetic. This allows for a distraction-free workspace where the focus remains on the image.

* **Surface**: Lighter Cream (`#FFFDD0`) for the active workspace.
* **Atmosphere**: Darker Cream (`#F3E5AB`) for the page background.
* **Typography**: Bold Noir (`#000000`) for headers, ensuring maximum readability.

---

## Tech Stack

* **Deep Learning Framework**: PyTorch
* **Acceleration**: MPS (Metal Performance Shaders) for Apple M-series GPUs
* **UI Framework**: Gradio 6.0
* **Image Processing**: OpenCV, PIL, NumPy

---

## Quick Start

1. **Clone the Repository**:
```bash
git clone https://github.com/pixelPanda123/Magic-Eraser.git

```


2. **Set Up Environment**:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

```


3. **Launch the Magic**:
```bash
python3 web_app.py

```



