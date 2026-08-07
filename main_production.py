"""Backward-compatible entrypoint for the Video Editing Copilot.

The maintained application now lives in :mod:`main`. This wrapper preserves
existing commands and imports without keeping a second drifting implementation.
"""

from main import VideoEditingCopilot, main

__all__ = ['VideoEditingCopilot', 'main']


if __name__ == '__main__':
    main()
