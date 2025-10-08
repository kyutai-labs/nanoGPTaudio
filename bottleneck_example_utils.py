import os
import subprocess
import tempfile
from typing import TypedDict

import numpy as np
import PIL
import PIL.Image
import PIL.ImageOps
import tqdm.auto
from einops import rearrange


class HistoryStep(TypedDict):
    samples: np.ndarray
    residuals: np.ndarray
    x_reconstructed: np.ndarray
    labels: np.ndarray
    codebooks: list[np.ndarray]
    codebooks_reconstructed: np.ndarray
    mse_loss: np.ndarray
    commitment_loss: np.ndarray


def paste_image_on_canvas(
    canvas: PIL.Image.Image, img: PIL.Image.Image, coord, min_xy, max_xy
):
    """
    Pastes a 28x28 image onto the canvas at the normalized coordinate.
    - img: PIL.Image.Image (already tinted and RGB)
    - coord: (x, y) original coordinates
    - min_xy: (min_x, min_y) for normalization
    - max_xy: (max_x, max_y) for normalization
    """
    assert img.width == img.height
    assert canvas.width == canvas.height
    # Normalize to [14, 512-14] to keep images (28x28) fully inside the canvas
    min_x, min_y = min_xy
    max_x, max_y = max_xy
    normalized = (np.array(coord) - [min_x, min_y]) / [max_x - min_x, max_y - min_y]
    # normalized = normalized * (512 - img.width) + img.width // 2
    normalized = normalized * canvas.width
    x, y = normalized
    x = int(x - img.width // 2)  # Center the image
    y = int(y - img.height // 2)
    canvas.paste(img, (x, y))


def image_for_history_step(
    history_step: HistoryStep, bounds: tuple[int, int, int, int] | None = None
):
    samples = history_step["samples"].copy()

    x_reconstructed = history_step["x_reconstructed"]
    images = np.clip(rearrange(x_reconstructed, "b (h w) -> b h w", h=28, w=28), 0, 1)

    if bounds is None:
        min_x, min_y = samples.min(axis=0)
        max_x, max_y = samples.max(axis=0)
    else:
        min_x, min_y, max_x, max_y = bounds

    canvas_size = 768
    canvas = PIL.Image.new("RGB", (canvas_size, canvas_size), 0)

    # Define a list of colors for tinting (RGB tuples)
    colors = [
        (100, 177, 166),
        (219, 219, 114),
        (157, 153, 183),
    ]

    for img_data, label, sample in zip(images, history_step["labels"], samples):
        assert img_data.dtype == np.float32
        img_data = (img_data * 255).astype(np.uint8)

        tint_color = colors[int(label) % len(colors)]

        # Convert grayscale image to RGB
        img = PIL.Image.fromarray(img_data, mode="L").convert("RGB")
        # Apply tint: blend with the tint color
        img = PIL.ImageOps.colorize(
            img.convert("L"), black="black", white="#%02x%02x%02x" % tint_color
        )

        # Paste tinted image onto canvas at normalized position
        paste_image_on_canvas(canvas, img, sample, (min_x, min_y), (max_x, max_y))

    codebook_images = np.clip(
        rearrange(
            history_step["codebooks_reconstructed"], "b (h w) -> b h w", h=28, w=28
        ),
        0,
        1,
    )

    for img_data, sample in zip(codebook_images, history_step["codebooks"][0]):
        assert img_data.dtype == np.float32
        img_data = (img_data * 255).astype(np.uint8)

        img = PIL.Image.fromarray(img_data, mode="L").convert("RGB")
        # Scale 2x
        img = img.resize(
            (
                int(img.width * 1.5),
                int(img.height * 1.5),
            ),
        )
        # Add a 1px white border
        img = PIL.ImageOps.expand(img, border=1, fill="white")

        paste_image_on_canvas(canvas, img, sample, (min_x, min_y), (max_x, max_y))

    return canvas


def export_history_to_mp4(history, output_path="history_steps.mp4", framerate=30):
    """
    Generates a PIL image for every history step and exports them as an MP4 using ffmpeg.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        bounds = (-1, -1, 1, 1)
        alpha = 0.9
        for i, step in tqdm.auto.tqdm(
            enumerate(history), total=len(history), desc="Generating frames"
        ):
            min_x, min_y = step["samples"].min(axis=0)
            max_x, max_y = step["samples"].max(axis=0)
            bounds_update = (min_x, min_y, max_x, max_y)
            updated = [
                alpha * p + (1 - alpha) * u for p, u in zip(bounds, bounds_update)
            ]
            bounds = tuple(updated)

            img = image_for_history_step(step, bounds)
            img.save(os.path.join(tmpdir, f"frame_{i:04d}.png"))

        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-framerate",
                str(framerate),
                "-i",
                os.path.join(tmpdir, "frame_%04d.png"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                output_path,
            ],
            check=True,
        )
    # The video is now saved at output_path
    return output_path
