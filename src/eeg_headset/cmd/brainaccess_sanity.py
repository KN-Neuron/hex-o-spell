"""Sanity check: try to connect to a real BrainAccess headset and report status.

Run when you have the BrainAccess MIDI plugged in (or paired via Bluetooth) to
verify the SDK can find it and stream a few seconds of data.

Usage:
    PYTHONPATH=. python -m src.eeg_headset.cmd.brainaccess_sanity \
        --model MIDI_16CH_BASE \
        --duration 5

What it does:
    1. Constructs a HeadsetConfig from the given model name + headsets.yaml.
    2. Tries to import and instantiate the BrainAccessDriver. If the
       brainaccess SDK is not installed, prints what to do and exits 2.
    3. Calls connect() — succeeds only if the OS sees the device with the
       device_name from headsets.yaml.
    4. Streams for `duration` seconds, polling every 100 ms.
    5. Reports total samples, mean amplitude per channel, and any obvious
       silent / saturated channels.

This is the script you run BEFORE attempting to launch the keyboard pipeline.
If it hangs or fails, the issue is hardware/driver level and the keyboard
won't work either.
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Optional

import numpy as np


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="MIDI_16CH_BASE",
        help="HeadsetModel name from headsets.yaml (default: MIDI_16CH_BASE)",
    )
    parser.add_argument(
        "--duration", type=float, default=5.0, help="Seconds to stream"
    )
    parser.add_argument(
        "--config",
        default="headsets.yaml",
        help="Path to headsets.yaml",
    )
    args = parser.parse_args(argv)

    # Import lazily so the script can report a sensible message if the SDK is missing.
    try:
        from src.eeg_headset.drivers.brainaccess import BrainAccessDriver
    except ImportError as e:
        print(f"❌ BrainAccess SDK not installed: {e}")
        print("Install it via the official BrainAccess wheel before running this.")
        return 2

    from src.eeg_headset.headset_config import HeadsetConfig, HeadsetModel

    try:
        model = HeadsetModel[args.model]
    except KeyError:
        print(f"❌ Unknown model '{args.model}'. Valid: {[m.name for m in HeadsetModel]}")
        return 1

    print(f"→ Building HeadsetConfig({args.model}) from {args.config}")
    cfg = HeadsetConfig(model=model, config_path=args.config)
    print(
        f"   device_name={cfg.device_name!r}, "
        f"n_channels={cfg.n_channels}, "
        f"sample_rate={cfg.sample_rate_hz} Hz"
    )

    print("→ Instantiating BrainAccessDriver…")
    driver = BrainAccessDriver(cfg)

    print(f"→ connect() — looking for device {cfg.device_name!r}…")
    try:
        driver.connect()
    except Exception as e:
        print(f"❌ connect() failed: {e}")
        print("Common causes:")
        print(f"  - Headset not powered on or not paired")
        print(f"  - device_name in headsets.yaml ({cfg.device_name!r}) doesn't match the actual device")
        print(f"  - BrainAccess SDK can't reach the OS Bluetooth stack")
        return 1
    print("✓ Connected.")

    print(f"→ start_stream() — streaming for {args.duration:.1f}s")
    driver.start_stream()

    collected: list[np.ndarray] = []
    deadline = time.monotonic() + args.duration
    while time.monotonic() < deadline:
        chunk = driver.read_available_samples()
        if chunk.size > 0:
            collected.append(chunk)
        time.sleep(0.1)

    driver.stop_stream()
    driver.disconnect()
    print("✓ Disconnected.")

    if not collected:
        print("❌ No samples received during the streaming window.")
        return 1

    data = np.concatenate(collected, axis=1)
    print(f"\n— Stream summary —")
    print(f"   samples   : {data.shape[1]} ({data.shape[1] / cfg.sample_rate_hz:.2f}s)")
    print(f"   channels  : {data.shape[0]}")
    print(f"   per-channel mean amplitude (µV):")
    means = np.mean(np.abs(data), axis=1)
    labels = list(cfg.channel_map.values())
    for i, (label, m) in enumerate(zip(labels, means)):
        flag = ""
        if m < 1.0:
            flag = "  ← suspiciously silent"
        elif m > 500.0:
            flag = "  ← saturated / disconnected?"
        print(f"     ch{i:2d} {label:>6s}  {m:7.2f}{flag}")

    print("\n✓ All good. You can now run the keyboard pipeline:")
    print("   PYTHONPATH=. python -m src.eeg_headset.cmd.run_keyboard --driver brainaccess")
    return 0


if __name__ == "__main__":
    sys.exit(main())
