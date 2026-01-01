# VoiceBatch Studio Pro - Init File
try:
    from importlib.metadata import version
except ImportError:
    from importlib_metadata import version  # For Python <3.8

# यहाँ हम वर्जन सेट कर रहे हैं
__version__ = "0.181.16-Turbo"

# यहाँ हमने नामों को बदलकर VoiceBatch कर दिया है
from .tts import VoiceBatchTTS
from .vc import VoiceBatchVC
from .tts_turbo import VoiceBatchTurboTTS
from .mtl_tts import VoiceBatchMultilingualTTS, SUPPORTED_LANGUAGES
