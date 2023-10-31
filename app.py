import os
from typing import Tuple, Union

import gradio as gr
import numpy as np
from PIL import Image, ImageChops, ImageOps


class ImageInfo:
    def __init__(
        self,
        size: Tuple[int, int],
        channels: int,
        data_type: str,
        min_val: float,
        max_val: float,
    ):
        self.size = size
        self.channels = channels
        self.data_type = data_type
        self.min_val = min_val
        self.max_val = max_val

    @classmethod
    def from_pil(cls, pil_image: Image.Image) -> "ImageInfo":
        size = (pil_image.width, pil_image.height)
        channels = len(pil_image.getbands())
        data_type = str(pil_image.mode)
        extrema = pil_image.getextrema()
        if channels > 1:  # Multi-band image
            min_val = min([band[0] for band in extrema])
            max_val = max([band[1] for band in extrema])
        else:  # Single-band image
            min_val, max_val = extrema
        return cls(size, channels, data_type, min_val, max_val)

    @classmethod
    def from_numpy(cls, np_array: np.ndarray) -> "ImageInfo":
        if len(np_array.shape) > 3:
            raise ValueError(f"Unsupported array shape: {np_array.shape}")
        size = (np_array.shape[1], np_array.shape[0])
        channels = 1 if len(np_array.shape) == 2 else np_array.shape[2]
        data_type = str(np_array.dtype)
        min_val, max_val = np_array.min(), np_array.max()
        return cls(size, channels, data_type, min_val, max_val)

    @classmethod
    def from_any(cls, image: Union[Image.Image, np.ndarray]) -> "ImageInfo":
        if isinstance(image, np.ndarray):
            return cls.from_numpy(image)
        elif isinstance(image, Image.Image):
            return cls.from_pil(image)
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

    def __str__(self) -> str:
        return f"{str(self.size)} {self.channels}C {self.data_type} {round(self.min_val, 2)}min/{round(self.max_val, 2)}max"

    @property
    def aspect_ratio(self) -> float:
        return self.size[0] / self.size[1]


def nextpow2(n):
    """Find the next power of 2 greater than or equal to `n`."""
    return int(2 ** np.ceil(np.log2(n)))


def pad_image_nextpow2(image):
    print("-" * 80)
    print("pad_image_nextpow2: ")
    print(ImageInfo.from_any(image))

    assert image.ndim in (2, 3), f"Expected 2D or 3D image. Got {image.ndim}D."

    height, width, channels = image.shape
    height_new = nextpow2(height)
    width_new = nextpow2(width)

    height_diff = height_new - height
    width_diff = width_new - width

    image = np.pad(
        image,
        (
            (height_diff // 2, height_diff - height_diff // 2),
            (width_diff // 2, width_diff - width_diff // 2),
            (0, 0),
        )
        if channels == 3
        else (
            (height_diff // 2, height_diff - height_diff // 2),
            (width_diff // 2, width_diff - width_diff // 2),
        ),
        mode="constant",
        # mode="edge",
        # mode="linear_ramp",
        # mode="maximum",
        # mode="mean",
        # mode="median",
        # mode="minimum",
        # mode="reflect",
        # mode="symmetric",
        # mode="wrap",
        # mode="empty",
    )

    print(ImageInfo.from_any(image))

    return image


def get_fft(image):
    print("-" * 80)
    print(f"get_fft: {image.shape}")
    print("image:", ImageInfo.from_any(image))

    fft = np.fft.fft2(image, axes=np.arange(image.ndim))
    fft = np.fft.fftshift(fft)

    return fft


def get_ifft_image(fft):
    print("-" * 80)
    print(f"get_ifft_image: {fft.shape}")

    ifft = np.fft.ifftshift(fft)
    ifft = np.fft.ifft2(ifft, axes=np.arange(fft.ndim))

    # we only need the real part
    ifft_image = np.real(ifft)

    # remove padding
    # ifft = ifft[
    #     h_diff // 2 : h_diff // 2 + original_shape[0],
    #     w_diff // 2 : w_diff // 2 + original_shape[1],
    # ]

    ifft_image = (ifft_image - np.min(ifft_image)) / (
        np.max(ifft_image) - np.min(ifft_image)
    )
    ifft_image = ifft_image * 255
    ifft_image = ifft_image.astype(np.uint8)

    return ifft_image


def fft_mag_image(fft):
    print("-" * 80)
    print(f"fft_mag_image: {fft.shape}")

    fft_mag = np.abs(fft)
    fft_mag = np.log(fft_mag + 1)

    # scale 0 to 1
    fft_mag = (fft_mag - np.min(fft_mag)) / (np.max(fft_mag) - np.min(fft_mag) + 1e-6)
    # scale to (0, 255)
    fft_mag = fft_mag * 255
    fft_mag = fft_mag.astype(np.uint8)
    return fft_mag


def fft_phase_image(fft):
    print("-" * 80)
    print(f"fft_phase_image: {fft.shape}")

    fft_phase = np.angle(fft)
    fft_phase = fft_phase + np.pi
    fft_phase = fft_phase / (2 * np.pi)

    # scale 0 to 1
    fft_phase = (fft_phase - np.min(fft_phase)) / (
        np.max(fft_phase) - np.min(fft_phase) + 1e-6
    )
    # scale to (0, 255)
    fft_phase = fft_phase * 255
    fft_phase = fft_phase.astype(np.uint8)
    return fft_phase


def onclick_process_fft(state, inp_image, mask_opacity, inverted_mask, pad):
    print("-" * 80)
    print("onclick_process_fft:")

    if isinstance(inp_image, dict):
        if "image" not in inp_image:
            raise gr.Error("Please upload or select an image first.")

        image, mask = inp_image["image"], inp_image["mask"]
        print("image:", ImageInfo.from_any(image))
        print("mask:", ImageInfo.from_any(image))

        image = Image.fromarray(image)
        mask = Image.fromarray(mask).convert(image.mode)

        if not inverted_mask:
            mask = ImageOps.invert(mask)

        image_final = ImageChops.multiply(image, mask)
        image_final = Image.blend(image, image_final, mask_opacity)

        image_final = image_final.convert(image.mode)
        image_final = np.array(image_final)
    elif isinstance(inp_image, np.ndarray):
        image_final = inp_image
    else:
        raise gr.Error("Please upload or select an image first.")

    print("image_final:", ImageInfo.from_any(image_final))

    if pad:
        image_final = pad_image_nextpow2(image_final)

    state["inp_image"] = image_final

    image_mag = fft_mag_image(get_fft(image_final))
    image_phase = fft_phase_image(get_fft(image_final))

    return (
        state,
        [
            (image_final, "Input Image (Final)"),
            (image_mag, "FFT Magnitude (Original)"),
            (image_phase, "FFT Phase (Original)"),
        ],
        image_mag,
        image_phase,
    )


def onclick_process_ifft(state, mag_and_mask, phase_and_mask):
    print("-" * 80)
    print("onclick_process_ifft:")
    if state["inp_image"] is None:
        raise gr.Error("Please process FFT first.")

    image = state["inp_image"]
    # h_new = nextpow2(original_shape[0])
    # w_new = nextpow2(original_shape[1])
    # h_diff = h_new - original_shape[0]
    # w_diff = w_new - original_shape[1]

    mask_mag = mag_and_mask["mask"]
    print("mag_mask:", ImageInfo.from_any(mask_mag))

    mask_phase = phase_and_mask["mask"]
    print("phase_mask:", ImageInfo.from_any(mask_phase))

    fft = get_fft(state["inp_image"])
    print(f"fft: {fft.shape}")

    fft_mag = np.where(mask_mag == 255, 0, np.abs(fft))
    fft_phase = np.where(mask_phase == 255, 0, np.angle(fft))

    fft = fft_mag * np.exp(1j * fft_phase)

    ifft_image = get_ifft_image(fft)
    image_mag = fft_mag_image(fft)
    image_phase = fft_phase_image(fft)

    return (
        [
            (image, "Input Image (Final)"),
            (image_mag, "FFT Magnitude (Final)"),
            (image_phase, "FFT Phase (Final)"),
        ],
        ifft_image,
    )


def get_start_image():
    return (np.ones((512, 512, 3)) * 255).astype(np.uint8)


def update_image_input(state, selection):
    print("-" * 80)
    print("update_image_input:")
    print(f"selection: {selection}")
    if not selection:
        white_image = get_start_image()
        return (
            white_image,
            [white_image],
            None,
            None,
            None,
        )

    image_path = os.path.join("./images", selection)
    print(f"image_path: {image_path}")
    if not os.path.exists(image_path):
        raise gr.Error(f"Image not found: {image_path}")

    image = Image.open(image_path)
    image = np.array(image)
    state["inp_image"] = image
    return (
        state,
        image,
        [image],
        None,
        None,
        None,
    )


def clear_image_input(state):
    print("-" * 80)
    print("clear_image_input:")
    state["inp_image"] = None
    return (
        state,
        None,
        [],
        None,
        None,
        None,
    )


css = """
.fft_mag > .image-container > button > div:first-child {
    display: none;
}
.fft_phase > .image-container > button > div:first-child {
    display: none;
}
.ifft_img > .image-container > button > div:first-child {
    display: none;
}
"""

with gr.Blocks(css=css) as demo:
    state = gr.State(
        {
            "inp_image": None,
        },
    )

    with gr.Row():
        with gr.Column():
            inp_image = gr.Image(
                value=get_start_image(),
                label="Input Image",
                height=512,
                type="numpy",
                interactive=True,
                tool="sketch",
                mask_opacity=1.0,
                elem_classes=["inp_img"],
            )
            files = os.listdir("./images")
            files = sorted(files)
            inp_samples = gr.Dropdown(
                choices=files,
                label="Select Example Image",
            )

        with gr.Column():
            out_gallery = gr.Gallery(
                label="Input Gallery",
                height=512,
                rows=1,
                columns=3,
                allow_preview=True,
                preview=False,
                selected_index=None,
            )

            with gr.Row():
                inp_mask_opacity = gr.Slider(
                    label="Mask Opacity",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    value=1.0,
                )

                inp_invert_mask = gr.Checkbox(
                    label="Invert Mask",
                    value=False,
                )

                inp_pad = gr.Checkbox(
                    label="Pad NextPow2",
                    value=True,
                )

    btn_fft = gr.Button("Process FFT")

    out_fft_mag = gr.Image(
        label="FFT Magnitude",
        height=512,
        type="numpy",
        interactive=True,
        # source="canvas",
        tool="sketch",
        mask_opacity=1.0,
        elem_classes=["fft_mag"],
    )
    out_fft_phase = gr.Image(
        label="FFT Phase",
        height=512,
        type="numpy",
        interactive=True,
        # source="canvas",
        tool="sketch",
        mask_opacity=1.0,
        elem_classes=["fft_phase"],
    )

    btn_ifft = gr.Button("Process IFFT")

    out_ifft = gr.Image(
        label="IFFT",
        height=512,
        type="numpy",
        interactive=True,
        show_download_button=True,
        elem_classes=["ifft_img"],
    )

    inp_image.clear(
        clear_image_input,
        [state],
        [state, inp_samples, out_gallery, out_fft_mag, out_fft_phase, out_ifft],
    )

    # Set up event listener for the Dropdown component to update the image input
    inp_samples.change(
        update_image_input,
        [state, inp_samples],
        [state, inp_image, out_gallery, out_fft_mag, out_fft_phase, out_ifft],
    )

    # Set up events for fft processing
    btn_fft.click(
        onclick_process_fft,
        [state, inp_image, inp_mask_opacity, inp_invert_mask, inp_pad],
        [state, out_gallery, out_fft_mag, out_fft_phase],
    )

    out_fft_mag.clear(
        onclick_process_fft,
        [state, inp_image, inp_mask_opacity, inp_invert_mask, inp_pad],
        [state, out_gallery, out_fft_mag, out_fft_phase],
    )

    out_fft_phase.clear(
        onclick_process_fft,
        [state, inp_image, inp_mask_opacity, inp_invert_mask, inp_pad],
        [state, out_gallery, out_fft_mag, out_fft_phase],
    )

    # inp_image.edit(
    #     get_fft_images,
    #     [state, inp_image],
    #     [out_gallery, out_fft_mag, out_fft_phase],
    # )

    # Set up events for ifft processing
    btn_ifft.click(
        onclick_process_ifft,
        [state, out_fft_mag, out_fft_phase],
        [out_gallery, out_ifft],
    )

    # out_fft_mag.edit(
    #     get_ifft_image,
    #     [state, out_fft_mag, out_fft_phase],
    #     [out_ifft],
    # )

    # out_fft_phase.edit(
    #     get_ifft_image,
    #     [state, out_fft_mag, out_fft_phase],
    #     [out_ifft],
    # )

if __name__ == "__main__":
    demo.launch()
