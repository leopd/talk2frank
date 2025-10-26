import numpy as np

from voice.tts_guts import TtsSynthesizer


def test_pick_speaker_prefers_list_order():
    synth = object.__new__(TtsSynthesizer)
    # inject dummy _tts
    class Dummy:
        speakers = ["p111", "p260", "p255", "p298"]
    synth._tts = Dummy()  # type: ignore[attr-defined]
    picked = TtsSynthesizer.pick_speaker(synth, synth._tts.speakers)  # type: ignore[arg-type]
    assert picked in ("p298", "p260", "p255")
    # ensure first match in preferred list wins
    speakers = ["foo", "p255", "p298"]
    picked2 = TtsSynthesizer.pick_speaker(synth, speakers)
    assert picked2 == "p298"


def test_process_audio_shapes_and_normalization():
    synth = object.__new__(TtsSynthesizer)
    # 1 second of dummy tone beyond range to exercise normalization
    sr = 16000
    t = np.linspace(0, 1, sr, endpoint=False)
    audio = (1.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    # configure with non-defaults to ensure parameters are used
    synth._pitch_down_steps = 7  # type: ignore[attr-defined]
    synth._distortion_gain = 1.8  # type: ignore[attr-defined]
    out = TtsSynthesizer._process_audio(synth, audio, sr)  # type: ignore[misc]
    assert out.dtype == np.float32
    assert out.shape == audio.shape
    assert np.max(np.abs(out)) <= 1.0001


def test_header_rate_reflects_slowdown_factor():
    # create dummy instance and monkeypatch writing to capture header sample rate
    synth = object.__new__(TtsSynthesizer)
    synth._pitch_down_steps = 5  # type: ignore[attr-defined]
    synth._distortion_gain = 1.4  # type: ignore[attr-defined]
    synth._output_sample_rate_factor = 0.3  # type: ignore[attr-defined]
    class Dummy:
        synthesizer = type("S", (), {"output_sample_rate": 16000})()
        def tts(self, text, speaker=None):
            return np.zeros(1600, dtype=np.float32)
        speakers = ["p298"]
    synth._tts = Dummy()  # type: ignore[attr-defined]

    data = synth.synthesize_wav("x")
    # RIFF header: bytes 24-27 contain sample rate little-endian
    rate = int.from_bytes(data[24:28], "little")
    assert rate == int(16000 * 0.3)


