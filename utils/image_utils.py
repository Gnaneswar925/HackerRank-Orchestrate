"""Safe, cost-conscious preparation of local images for vision requests."""

from __future__ import annotations

import base64
import hashlib
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError


Image.MAX_IMAGE_PIXELS = 40_000_000

MAX_SOURCE_BYTES = 25 * 1024 * 1024
MAX_OUTPUT_BYTES = 8 * 1024 * 1024
DEFAULT_MAX_DIMENSION = 2048
SUPPORTED_FORMATS = frozenset({"JPEG", "PNG", "WEBP"})


class ImageProcessingError(ValueError):
    """Raised when local evidence cannot be safely decoded and prepared."""


@dataclass(frozen=True, slots=True)
class PreparedImage:
    """A normalized image ready to be sent as an OpenAI data URL."""

    image_id: str
    source_path: str
    data_url: str
    width: int
    height: int
    sha256: str


def prepare_image(
    image_path: str | Path,
    image_id: str,
    *,
    max_dimension: int = DEFAULT_MAX_DIMENSION,
) -> PreparedImage:
    """Validate, orient, resize, and JPEG-encode one local image.

    Resizing bounds request cost while EXIF transposition prevents accidental
    sideways analysis. The original file is never modified.
    """

    path = Path(image_path).expanduser()
    if not image_id.strip():
        raise ValueError("image_id must be non-empty")
    if max_dimension < 256:
        raise ValueError("max_dimension must be at least 256 pixels")
    if not path.exists():
        raise ImageProcessingError(f"image does not exist: {path}")
    if not path.is_file():
        raise ImageProcessingError(f"image path is not a file: {path}")
    if path.stat().st_size > MAX_SOURCE_BYTES:
        raise ImageProcessingError(
            f"image exceeds the {MAX_SOURCE_BYTES // (1024 * 1024)} MB limit"
        )

    try:
        with Image.open(path) as opened:
            if opened.format not in SUPPORTED_FORMATS:
                allowed = ", ".join(sorted(SUPPORTED_FORMATS))
                raise ImageProcessingError(f"unsupported image format; use {allowed}")
            opened.load()
            image = ImageOps.exif_transpose(opened)
            image.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
            image = _to_rgb(image)
            encoded = _encode_with_size_limit(image)
    except ImageProcessingError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageProcessingError(f"unreadable or corrupt image: {path}") from exc
    except Image.DecompressionBombError as exc:
        raise ImageProcessingError("image dimensions exceed the safety limit") from exc

    digest = hashlib.sha256(encoded).hexdigest()
    data = base64.b64encode(encoded).decode("ascii")
    return PreparedImage(
        image_id=image_id,
        source_path=str(path),
        data_url=f"data:image/jpeg;base64,{data}",
        width=image.width,
        height=image.height,
        sha256=digest,
    )


def _to_rgb(image: Image.Image) -> Image.Image:
    if image.mode in {"RGBA", "LA"} or (
        image.mode == "P" and "transparency" in image.info
    ):
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, "white")
        return Image.alpha_composite(background, rgba).convert("RGB")
    return image.convert("RGB")


def _encode_with_size_limit(image: Image.Image) -> bytes:
    working = image
    for quality in (88, 80, 72, 64):
        buffer = BytesIO()
        working.save(buffer, format="JPEG", quality=quality, optimize=True)
        encoded = buffer.getvalue()
        if len(encoded) <= MAX_OUTPUT_BYTES:
            return encoded
    raise ImageProcessingError(
        f"prepared image exceeds the {MAX_OUTPUT_BYTES // (1024 * 1024)} MB limit"
    )
