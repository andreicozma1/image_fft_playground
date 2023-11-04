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

    assert image.ndim in (
        2,
        3,
    ), f"Expected (H, W) or (H, W, C) image, got {image.shape}"

    height, width, channels = image.shape
    height_new = nextpow2(height)
    width_new = nextpow2(width)

    height_diff = height_new - height
    width_diff = width_new - width

    # Determine the padding for each dimension
    padding = (
        (height_diff // 2, height_diff - height_diff // 2),  # Padding for height
        (width_diff // 2, width_diff - width_diff // 2),  # Padding for width
    )

    if image.ndim == 3:
        padding += ((0, 0),)  # No padding for channels

    # Pad the image
    image_padded = np.pad(
        image,
        padding,
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


def scale_0_to_1(array: np.ndarray) -> np.ndarray:
    array = (array - np.min(array)) / (np.max(array) - np.min(array) + 1e-6)
    return array


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

    ifft_image = scale_0_to_1(ifft_image)
    ifft_image = ifft_image * 255
    ifft_image = ifft_image.astype(np.uint8)
    return ifft_image


def fft_mag_image(fft):
    print("-" * 80)
    print(f"fft_mag_image: {fft.shape}")

    fft_mag = np.abs(fft)
    fft_mag = np.log(fft_mag + 1)

    fft_mag = scale_0_to_1(fft_mag)
    fft_mag = fft_mag * 255
    fft_mag = fft_mag.astype(np.uint8)
    return fft_mag


def fft_phase_image(fft):
    print("-" * 80)
    print(f"fft_phase_image: {fft.shape}")

    fft_phase = np.angle(fft)
    # fft_phase = fft_phase + np.pi
    # fft_phase = fft_phase / (2 * np.pi)

    fft_phase = scale_0_to_1(fft_phase)
    fft_phase = fft_phase * 255
    fft_phase = fft_phase.astype(np.uint8)
    return fft_phase
