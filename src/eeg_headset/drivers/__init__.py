from .headset_driver import HeadsetDriver
from .mock import MockDriver
from .playback import PlaybackDriver

try:
    from .brainaccess import BrainAccessDriver
except ImportError:
    print("Warning: 'brainaccess' module not found. BrainAccessDriver will not be available.")
    pass

try:
    from .bioamp import BioAmpEXGDriver, BioAmpConfig
except ImportError:
    # pyserial not installed — BioAmp driver remains unavailable.
    # Don't print a warning here; only matters if the user wires --bioamp-port.
    pass