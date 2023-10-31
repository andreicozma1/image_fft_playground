import os

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter


def create_blank_image(size=256):
    return np.zeros((size, size), dtype=np.uint8)


def add_horizontal_lines(image, spacing=10):
    for i in range(0, image.shape[0], spacing):
        image[i, :] = 255
    return image


def add_vertical_lines(image, spacing=10):
    for i in range(0, image.shape[1], spacing):
        image[:, i] = 255
    return image


def add_diagonal_lines(image, spacing=10):
    for i in range(0, image.shape[0], spacing):
        np.fill_diagonal(image[i:, i:], 255)
    return image


# Add a circle to the image
def add_circle(image, radius=50):
    center_x, center_y = image.shape[1] // 2, image.shape[0] // 2
    y, x = np.ogrid[
        -center_y : image.shape[0] - center_y, -center_x : image.shape[1] - center_x
    ]
    mask = x * x + y * y <= radius * radius
    image[mask] = 255
    return image


def add_checkerboard(image, square_size=16):
    for i in range(0, image.shape[0], square_size * 2):
        for j in range(0, image.shape[1], square_size * 2):
            image[i : i + square_size, j : j + square_size] = 255
            image[
                i + square_size : i + square_size * 2,
                j + square_size : j + square_size * 2,
            ] = 255
    return image


def add_horizontal_sinusoidal(image, frequency=1 / 20, amplitude=127):
    x = np.linspace(0, 1, image.shape[1])
    y = np.sin(2 * np.pi * frequency * x) * amplitude + 128
    for i in range(image.shape[0]):
        image[i, :] = y
    return image


def add_vertical_sinusoidal(image, frequency=1 / 20, amplitude=127):
    x = np.linspace(0, 1, image.shape[0])
    y = np.sin(2 * np.pi * frequency * x) * amplitude + 128
    for i in range(image.shape[1]):
        image[:, i] = y
    return image


def add_random_noise(image, std_dev=50):
    noise = np.random.normal(0, std_dev, image.shape).astype(np.float32)
    image = image.astype(np.float32) + noise
    np.clip(image, 0, 255, out=image)
    image = image.astype(np.uint8)
    return image


def save_image(image, filename):
    image = Image.fromarray(image)
    image.save(filename)


savedir = "./images/"

os.makedirs(savedir, exist_ok=True)

save_image(create_blank_image(), savedir + "blank.png")
save_image(add_horizontal_lines(create_blank_image()), savedir + "horizontal.png")
save_image(add_vertical_lines(create_blank_image()), savedir + "vertical.png")
save_image(add_diagonal_lines(create_blank_image()), savedir + "diagonal.png")
save_image(add_circle(create_blank_image()), savedir + "circle.png")
save_image(add_checkerboard(create_blank_image()), savedir + "checkerboard.png")
save_image(
    add_horizontal_sinusoidal(create_blank_image()),
    savedir + "horizontal_sinusoidal.png",
)
save_image(
    add_vertical_sinusoidal(create_blank_image()), savedir + "vertical_sinusoidal.png"
)
save_image(add_random_noise(create_blank_image()), savedir + "random_noise.png")
