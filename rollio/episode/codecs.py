"""Codec discovery and encoding presets for Rollio exports."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from functools import lru_cache

import av
import numpy as np


@dataclass(frozen=True)
class CodecOption:
    """A user-selectable export codec."""

    name: str
    label: str
    kind: str
    pyav_codec: str
    container_format: str
    file_extension: str
    input_pixel_format: str
    stream_pixel_format: str | None = None
    codec_options: dict[str, str] = field(default_factory=dict)


RGB_CODEC_OPTIONS: tuple[CodecOption, ...] = (
    CodecOption(
        name="h264_nvenc",
        label="H.264 (NVIDIA NVENC)",
        kind="rgb",
        pyav_codec="h264_nvenc",
        container_format="mp4",
        file_extension=".mp4",
        input_pixel_format="bgr24",
        stream_pixel_format="yuv420p",
        codec_options={"preset": "p4", "cq": "19"},
    ),
    CodecOption(
        name="libx264",
        label="H.264 (libx264)",
        kind="rgb",
        pyav_codec="libx264",
        container_format="mp4",
        file_extension=".mp4",
        input_pixel_format="bgr24",
        stream_pixel_format="yuv420p",
        codec_options={"preset": "veryfast", "crf": "18"},
    ),
    CodecOption(
        name="mpeg4",
        label="MPEG-4",
        kind="rgb",
        pyav_codec="mpeg4",
        container_format="mp4",
        file_extension=".mp4",
        input_pixel_format="bgr24",
        stream_pixel_format="yuv420p",
        codec_options={"q:v": "2"},
    ),
)

DEPTH_CODEC_OPTIONS: tuple[CodecOption, ...] = (
    CodecOption(
        name="ffv1",
        label="FFV1 (lossless)",
        kind="depth",
        pyav_codec="ffv1",
        container_format="matroska",
        file_extension=".mkv",
        input_pixel_format="gray16le",
        stream_pixel_format="gray16le",
    ),
    CodecOption(
        name="rawvideo",
        label="Raw video (lossless, large)",
        kind="depth",
        pyav_codec="rawvideo",
        container_format="matroska",
        file_extension=".mkv",
        input_pixel_format="gray16le",
        stream_pixel_format="gray16le",
    ),
)

RGB_CODEC_ALIASES = {"mp4v": "mpeg4"}
DEPTH_CODEC_ALIASES = {"raw": "rawvideo"}


def _probe_codec_option(codec: CodecOption) -> bool:
    """Return whether a codec/container pair can encode one tiny frame."""
    if codec.kind == "rgb":
        frame_data = np.zeros((16, 16, 3), dtype=np.uint8)
    else:
        frame_data = np.zeros((16, 16), dtype=np.uint16)
    with tempfile.NamedTemporaryFile(suffix=codec.file_extension) as probe_file:
        try:
            with av.open(probe_file.name, mode="w", format=codec.container_format) as out:
                stream = out.add_stream(
                    codec.pyav_codec,
                    rate=1,
                    options=codec.codec_options or None,
                )
                stream.width = int(frame_data.shape[1])
                stream.height = int(frame_data.shape[0])
                if codec.stream_pixel_format:
                    stream.pix_fmt = codec.stream_pixel_format
                frame = av.VideoFrame.from_ndarray(
                    frame_data,
                    format=codec.input_pixel_format,
                )
                for packet in stream.encode(frame):
                    out.mux(packet)
                for packet in stream.encode():
                    out.mux(packet)
            return True
        except Exception:  # pragma: no cover - probe failures are environment-dependent
            return False


@lru_cache(maxsize=1)
def available_rgb_codec_options() -> tuple[CodecOption, ...]:
    """Return usable RGB codec options ordered by preference."""
    available: list[CodecOption] = []
    for option in RGB_CODEC_OPTIONS:
        if _probe_codec_option(option):
            available.append(option)
    return tuple(available or (RGB_CODEC_OPTIONS[-1],))


@lru_cache(maxsize=1)
def available_depth_codec_options() -> tuple[CodecOption, ...]:
    """Return usable depth codec options ordered by preference."""
    available: list[CodecOption] = []
    for option in DEPTH_CODEC_OPTIONS:
        if _probe_codec_option(option):
            available.append(option)
    return tuple(available or (DEPTH_CODEC_OPTIONS[0],))


def default_rgb_codec_name() -> str:
    """Return the preferred RGB codec name for this machine."""
    return available_rgb_codec_options()[0].name


def default_depth_codec_name() -> str:
    """Return the preferred depth codec name for this machine."""
    return available_depth_codec_options()[0].name


def _normalize_codec_name(name: str, aliases: dict[str, str]) -> str:
    return aliases.get(name, name)


def get_rgb_codec_option(name: str) -> CodecOption:
    """Resolve one RGB codec option by configured name."""
    normalized = _normalize_codec_name(name, RGB_CODEC_ALIASES)
    for option in RGB_CODEC_OPTIONS:
        if option.name == normalized:
            return option
    raise KeyError(f"Unknown RGB codec: {name}")


def get_depth_codec_option(name: str) -> CodecOption:
    """Resolve one depth codec option by configured name."""
    normalized = _normalize_codec_name(name, DEPTH_CODEC_ALIASES)
    for option in DEPTH_CODEC_OPTIONS:
        if option.name == normalized:
            return option
    raise KeyError(f"Unknown depth codec: {name}")
