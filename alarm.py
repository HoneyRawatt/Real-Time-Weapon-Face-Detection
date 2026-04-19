"""
Audio alarm module.

pygame is initialised lazily on the first call to start_alarm() so that
importing this module on a headless server (no audio device) does not
crash the entire application at startup.
"""
import logging

logger = logging.getLogger(__name__)

_mixer_ready  = False
_alarm_sound  = None


def _init_mixer() -> bool:
    """Initialise pygame mixer on first use.  Returns True if ready."""
    global _mixer_ready, _alarm_sound
    if _mixer_ready:
        return True
    try:
        import pygame
        pygame.mixer.init()
        _alarm_sound = pygame.mixer.Sound("alarm.wav")
        _mixer_ready = True
        logger.info("Audio mixer initialised (alarm.wav loaded).")
    except Exception as exc:
        logger.warning(f"Audio not available — alarm will be silent. Reason: {exc}")
        _mixer_ready = False
    return _mixer_ready


def start_alarm() -> None:
    if _init_mixer() and _alarm_sound is not None:
        _alarm_sound.play(maxtime=5000)
        logger.info("Alarm started.")


def stop_alarm() -> None:
    if _mixer_ready and _alarm_sound is not None:
        _alarm_sound.stop()
        logger.info("Alarm stopped.")
