"""Tests for LeRobot episode export writing."""

from __future__ import annotations

from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
av = pytest.importorskip("av")

import rollio.episode.writer as writer_module
from rollio.episode.codecs import get_depth_codec_option
from rollio.episode.writer import LeRobotV21Writer


def test_writer_raises_clean_error_on_pyav_open_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_open(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("boom")

    monkeypatch.setattr(writer_module.av, "open", fake_open)

    writer = LeRobotV21Writer(
        root=tmp_path,
        project_name="pyav_error_guard",
        fps=10,
        camera_configs={},
        video_codec="libx264",
        depth_codec="ffv1",
    )
    frames = [(0.0, np.zeros((32, 32), dtype=np.uint16))]
    codec = get_depth_codec_option("ffv1")

    with pytest.raises(RuntimeError, match="PyAV failed while encoding"):
        writer._write_video(  # pylint: disable=protected-access
            tmp_path / "episode_000000.mkv",
            frames,
            10,
            codec,
        )


def test_writer_encodes_depth_gray16le_frames(tmp_path: Path) -> None:
    writer = LeRobotV21Writer(
        root=tmp_path,
        project_name="depth_encode",
        fps=10,
        camera_configs={},
        video_codec="libx264",
        depth_codec="ffv1",
    )
    frames = [
        (0.0, np.zeros((24, 24), dtype=np.uint16)),
        (0.1, np.ones((24, 24), dtype=np.uint16) * 1200),
    ]
    codec = get_depth_codec_option("ffv1")
    out_path = tmp_path / "episode_000000.mkv"

    writer._write_video(  # pylint: disable=protected-access
        out_path,
        frames,
        10,
        codec,
    )

    assert out_path.exists()
    assert out_path.stat().st_size > 0
