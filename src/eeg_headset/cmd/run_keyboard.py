"""Main entry point for the BrainBoard keyboard pipeline.

This is what you run when you want the full system: EEG → MI classifier +
blink detector → speller. Driver is selected via CLI flag.

Usage examples:
    # Demo with synthetic data, no hardware:
    PYTHONPATH=. python -m src.eeg_headset.cmd.run_keyboard --driver mock

    # Replay from a saved .npy:
    PYTHONPATH=. python -m src.eeg_headset.cmd.run_keyboard \
        --driver playback --playback-source data/X.npy --headset-model SAMPLE_64CH

    # Real hardware: BrainAccess MIDI as both EEG and EOG source:
    PYTHONPATH=. python -m src.eeg_headset.cmd.run_keyboard \
        --driver brainaccess --headset-model MIDI_16CH_BASE

    # Real hardware with separate BioAmp EOG (Etap 5):
    PYTHONPATH=. python -m src.eeg_headset.cmd.run_keyboard \
        --driver brainaccess --headset-model MIDI_16CH_BASE \
        --bioamp-port /dev/ttyUSB0 --bioamp-baud 115200

Channel selection for blink detection is automatic:
    - With --bioamp-port: blink reads from the BioAmp stream (channel 0).
    - Without: blink reads Fp1/Fp2 from the main EEG (indexes 0,1 in MIDI_16CH_BASE).
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional

from src.eeg_headset.blink_detector import (
    BlinkDetector,
    BlinkDetectorConfig,
    BlinkSource,
)
from src.eeg_headset.eeg_headset import EEGHeadset
from src.eeg_headset.headset_config import HeadsetConfig, HeadsetModel
from src.inference.pipeline import BCIPipeline, PipelineConfig
from src.inference.starter_bci import load_model
from src.speller import Speller
from src.speller.state import SpellerStateIdle


def _build_main_driver(args: argparse.Namespace, config: HeadsetConfig):
    """Construct the headset driver requested via --driver."""
    from src.eeg_headset.drivers import MockDriver, PlaybackDriver

    if args.driver == "mock":
        return MockDriver(config=config)

    if args.driver == "playback":
        if not args.playback_source:
            raise SystemExit("--driver playback requires --playback-source PATH")
        return PlaybackDriver(config=config, source=args.playback_source, loop=True)

    if args.driver == "scripted":
        from src.eeg_headset.drivers import Intent, ScriptedDriver
        if not args.scripted_sequence:
            raise SystemExit(
                "--driver scripted requires --scripted-sequence "
                "(comma-separated, e.g. 'blink,right,right,blink')"
            )
        try:
            intents = [Intent[s.strip().upper()] for s in args.scripted_sequence.split(",")]
        except KeyError as e:
            valid = [i.name for i in Intent]
            raise SystemExit(f"Invalid intent {e}. Valid: {valid}")
        return ScriptedDriver(config, intents)

    if args.driver == "brainaccess":
        try:
            from src.eeg_headset.drivers.brainaccess import BrainAccessDriver
        except ImportError as e:
            raise SystemExit(
                f"BrainAccess SDK not installed: {e}\n"
                "Install the brainaccess wheel before using --driver brainaccess."
            )
        return BrainAccessDriver(config)

    raise SystemExit(f"Unknown driver: {args.driver}")


def _build_eog_headset(args: argparse.Namespace) -> Optional[EEGHeadset]:
    """If --bioamp-port is given, build a separate EEGHeadset for the BioAmp stream.

    Returns None if EOG should come from the main EEG headset (Fp1/Fp2 fallback).
    """
    if not args.bioamp_port:
        return None

    from src.eeg_headset.drivers.bioamp import BioAmpEXGDriver, BioAmpConfig

    bioamp_cfg = BioAmpConfig(
        port=args.bioamp_port,
        baudrate=args.bioamp_baud,
        sample_rate_hz=args.bioamp_sfreq,
    )
    driver = BioAmpEXGDriver(bioamp_cfg)
    return EEGHeadset(driver, buffer_size_seconds=10)


def _build_blink_detector(args: argparse.Namespace) -> BlinkDetector:
    """Pick blink source (BioAmp ch 0 vs Fp1/Fp2) based on whether BioAmp is wired in."""
    if args.bioamp_port:
        return BlinkDetector(
            source=BlinkSource.DEDICATED_EOG,
            frontal_channel_idx=[0],  # BioAmp single channel
        )
    # Default: prefrontal EEG. For MIDI_16CH_BASE, Fp1/Fp2 are at indices 0, 1.
    return BlinkDetector(
        source=BlinkSource.PREFRONTAL_EEG,
        frontal_channel_idx=[0, 1],
    )


def _build_layout(name: str):
    """Construct the speller layout requested via --layout."""
    from src.speller import BigramAdaptiveLayout, StaticGridLayout
    if name == "bigram":
        return BigramAdaptiveLayout()
    return StaticGridLayout()


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    # Main EEG driver
    parser.add_argument(
        "--driver",
        choices=["mock", "playback", "brainaccess", "scripted"],
        default="mock",
        help="Where to source EEG data from",
    )
    parser.add_argument(
        "--scripted-sequence",
        default=None,
        help="(--driver scripted only) comma-separated intents, e.g. 'blink,blink,right,right,blink' "
        "(LEFT/RIGHT/BLINK/REST, case-insensitive)",
    )
    parser.add_argument(
        "--layout",
        choices=["static", "bigram"],
        default="static",
        help="Speller layout (default: static = original 5×6, bigram = adaptive Polish ring)",
    )
    parser.add_argument(
        "--headset-model",
        default="MIDI_16CH_BASE",
        help="HeadsetModel name (must match a key in headsets.yaml)",
    )
    parser.add_argument(
        "--headsets-yaml",
        default="headsets.yaml",
        help="Path to headsets.yaml",
    )
    parser.add_argument(
        "--playback-source",
        default=None,
        help="(--driver playback only) path to .npy with EEG data",
    )

    # Optional BioAmp EOG (Etap 5)
    parser.add_argument(
        "--bioamp-port",
        default=None,
        help="Serial port for BioAmp EXG Pill (e.g. /dev/ttyUSB0). "
        "If unset, blink reads from prefrontal EEG channels.",
    )
    parser.add_argument(
        "--bioamp-baud", type=int, default=115200, help="BioAmp serial baud rate"
    )
    parser.add_argument(
        "--bioamp-sfreq",
        type=int,
        default=500,
        help="BioAmp ADC sample rate (matches the firmware on your ESP/Arduino)",
    )

    # Model
    parser.add_argument(
        "--model-path",
        default="data/model/final_best.pth",
        help="Path to EEGNet checkpoint (.pth state_dict)",
    )

    # Pipeline tuning
    parser.add_argument("--confidence-cutoff", type=float, default=0.55)
    parser.add_argument("--smoothing-window", type=int, default=5)
    parser.add_argument("--min-votes-required", type=int, default=3)
    parser.add_argument("--epoch-seconds", type=int, default=4)
    parser.add_argument("--min-blink-interval", type=float, default=1.5)

    # Preprocessing — must match what the model was trained with.
    parser.add_argument("--preprocess-sfreq", type=float, default=160.0)
    parser.add_argument("--preprocess-bandpass-low", type=float, default=0.0)
    parser.add_argument("--preprocess-bandpass-high", type=float, default=49.0)

    args = parser.parse_args(argv)

    # ─── Build everything ───────────────────────────────────────────────────
    print(f"→ Loading HeadsetModel.{args.headset_model} from {args.headsets_yaml}")
    config = HeadsetConfig(
        model=HeadsetModel[args.headset_model], config_path=args.headsets_yaml
    )

    print(f"→ Building driver: {args.driver}")
    driver = _build_main_driver(args, config)
    headset = EEGHeadset(driver, buffer_size_seconds=10)

    eog_headset = _build_eog_headset(args)
    if eog_headset:
        print(f"→ BioAmp EOG enabled on {args.bioamp_port}")

    print(f"→ Loading model from {args.model_path}")
    model = load_model(args.model_path)

    typed_buffer = []

    def _on_letter(letter: str) -> None:
        if letter == "·":
            return
        typed_buffer.append(letter)
        print(f"\n[TEXT] {''.join(typed_buffer)}")

    speller = Speller(
        layout=_build_layout(args.layout),
        on_letter_select=_on_letter
    )
    speller.state = SpellerStateIdle()

    blink_detector = _build_blink_detector(args)

    pipeline_cfg = PipelineConfig(
        confidence_cutoff=args.confidence_cutoff,
        smoothing_window=args.smoothing_window,
        min_votes_required=args.min_votes_required,
        epoch_seconds=args.epoch_seconds,
        min_blink_interval_s=args.min_blink_interval,
        preprocess_sfreq=args.preprocess_sfreq,
        preprocess_bandpass_low=args.preprocess_bandpass_low,
        preprocess_bandpass_high=args.preprocess_bandpass_high,
    )

    def _on_action(action: str) -> None:
        print(f"   [action] {action}")

    pipeline = BCIPipeline(
        model=model,
        headset=headset,
        speller=speller,
        blink_detector=blink_detector,
        eog_headset=eog_headset,
        config=pipeline_cfg,
        on_action=_on_action,
    )

    pipeline.start()
    pipeline.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
