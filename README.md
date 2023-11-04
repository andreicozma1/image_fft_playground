---
title: Image Fft Playground
emoji: 📊
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 3.50.2
app_file: app.py
pinned: false
license: apache-2.0
---

# Image FFT Playground

Gradio app to play with FFT of images.

Huggingface Spaces Demo: <https://huggingface.co/spaces/acozma/image_fft_playground>

## Usage

- Upload your own input image, or use one of the example images
- Optionally draw on the input image to see how the FFT changes
- Mask out areas of the FFT magnitude and phase to see how the image changes
- Apply IFFT to the modified FFT to see the result

## Example Screenshots

![Generate FFT](image.png)

![Generate IFFT](image-1.png)

## Known Bugs

- Gradio Image canvas sometimes bugs out and unintentionally resizes/crops the images on the FFT Magnitude and Phase canvases. As a result, taking the IFFT will produce an error. To fix this, just refresh the page and try again.
  - In the process of re-writing the app using Streamlit, since Gradio image canvas is not very stable and the API is not very flexible.
