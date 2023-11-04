import os

import numpy as np
import streamlit as st
from PIL import Image, ImageChops, ImageOps
from streamlit_drawable_canvas import st_canvas

from utils import *


def process_fft(inp_image, pad):
    if inp_image is None:
        raise ValueError("inp_image must not be None.")

    if pad:
        inp_image = pad_image_nextpow2(inp_image)

    image_fft = get_fft(inp_image)

    return image_fft


def process_ifft(fft, mag_canvas, phase_canvas):
    if mag_canvas is None or phase_canvas is None:
        raise ValueError("mag_canvas and phase_canvas must not be None.")

    if fft is None:
        raise ValueError("fft must not be None.")

    # Normalize the drawn modifications to a range compatible with the FFT
    fft_mag = np.abs(fft)
    fft_phase = np.angle(fft)

    mag_canvas = Image.fromarray(mag_canvas.astype(np.uint8))
    phase_canvas = Image.fromarray(phase_canvas.astype(np.uint8))

    # apply the drawn modifications to the FFT
    # mag_canvas = mag_canvas.resize(fft_mag.shape[:2][::-1])
    # phase_canvas = phase_canvas.resize(fft_phase.shape[:2][::-1])

    # mag_canvas = np.array(mag_canvas)
    # phase_canvas = np.array(phase_canvas)

    # fft_mag = fft_mag * (mag_canvas / 255)
    # fft_phase = fft_phase * (phase_canvas / 255)

    # reconstruct the image from the modified FFT
    fft = fft_mag * np.exp(1j * fft_phase)

    return get_ifft_image(fft)


def main():
    st.set_page_config(layout="wide")

    st.title("Image FFT and IFFT Processor")

    st.sidebar.header("Input Image Controls")
    input_drawing_mode = st.sidebar.selectbox(
        "Input Drawing tool:",
        ("freedraw", "line", "rect", "circle", "transform"),
        key="input_drawing_mode",
    )
    input_stroke_width = st.sidebar.slider(
        "Input Stroke width: ",
        1,
        25,
        3,
        key="input_stroke_width",
    )

    input_stroke_color = st.sidebar.color_picker(
        "Input Stroke Color",
        "#FFFFFF",
        key="input_stroke_color",
    )
    input_fill_color = st.sidebar.color_picker(
        "Input Fill Color",
        "#000000",
        key="input_fill_color",
    )

    example_images = sorted(os.listdir("./images"))
    uploaded_file = st.sidebar.file_uploader(
        "Upload Image", type=["png", "jpg", "jpeg"]
    )
    selected_example = st.sidebar.selectbox(
        "or Select Example Image", ["None"] + example_images
    )

    # FFT Magnitude/Phase Drawing Controls
    st.sidebar.header("FFT Magnitude/Phase Drawing Controls")
    fft_drawing_mode = st.sidebar.selectbox(
        "FFT Drawing tool:",
        ("freedraw", "line", "rect", "circle", "transform"),
        key="fft_drawing_mode",
    )
    fft_stroke_width = st.sidebar.slider(
        "FFT Stroke width: ",
        1,
        25,
        8,
        key="fft_stroke_width",
    )

    if uploaded_file is not None:
        bg_image = Image.open(uploaded_file).convert("RGBA")
    elif selected_example != "None":
        bg_image_path = os.path.join("./images", selected_example)
        bg_image = Image.open(bg_image_path).convert("RGBA")
    else:
        bg_image = Image.new("RGBA", (512, 512), input_fill_color)

    io_cols = st.columns(2)
    with io_cols[0]:
        st.header("Canvas")

        canvas_result = st_canvas(
            stroke_width=input_stroke_width,
            stroke_color=input_stroke_color,
            background_image=bg_image,
            update_streamlit=True,
            width=bg_image.width,
            height=bg_image.height,
            drawing_mode=input_drawing_mode,
            key="canvas",
        )

    if canvas_result is None or canvas_result.image_data is None:
        st.error("Waiting for Canvas to be initialized.")
        return

    drawn_image = Image.fromarray(canvas_result.image_data.astype(np.uint8)).convert(
        "RGBA"
    )
    # Resize drawn_image to match the dimensions of bg_image if necessary
    if drawn_image.size != bg_image.size:
        drawn_image = drawn_image.resize(bg_image.size)
    input_image = Image.alpha_composite(bg_image, drawn_image)

    input_image = np.array(input_image)
    image_fft = process_fft(
        input_image,
        pad=True,
    )

    fft_cols = st.columns(2)

    with fft_cols[0]:
        st.header("FFT Magnitude")
        image_mag = Image.fromarray(fft_mag_image(image_fft))
        mag_canvas = st_canvas(
            fill_color="#FFFFFF",
            stroke_width=fft_stroke_width,
            stroke_color="#000000",
            background_image=image_mag,
            update_streamlit=True,
            width=image_mag.width,
            height=image_mag.height,
            drawing_mode=fft_drawing_mode,
            key="mag_canvas",
        )
        st.write("FFT Magnitude")
        st.write(ImageInfo.from_any(image_mag))

    with fft_cols[1]:
        st.header("FFT Magnitude")
        image_phase = Image.fromarray(fft_phase_image(image_fft))
        phase_canvas = st_canvas(
            fill_color="#FFFFFF",
            stroke_width=fft_stroke_width,
            stroke_color="#000000",
            background_image=image_phase,
            update_streamlit=True,
            width=image_phase.width,
            height=image_phase.height,
            drawing_mode=fft_drawing_mode,
            key="phase_canvas",
        )
        st.write("FFT Phase")
        st.write(ImageInfo.from_any(image_phase))

    with io_cols[1]:
        st.header("IFFT Image")
        if mag_canvas is None or mag_canvas.image_data is None:
            st.error("Waiting for FFT Magnitude Canvas to be initialized.")
            return
        if phase_canvas is None or phase_canvas.image_data is None:
            st.error("Waiting for FFT Phase Canvas to be initialized.")
            return
        ifft_image = process_ifft(
            image_fft, mag_canvas.image_data, phase_canvas.image_data
        )
        st.image(
            ifft_image,
            caption=ImageInfo.from_any(ifft_image),
        )


if __name__ == "__main__":
    main()
