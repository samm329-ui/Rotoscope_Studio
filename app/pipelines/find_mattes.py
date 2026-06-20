# Snake-case alias for FindMattes.py so the fast matte pipeline
# can be imported using lowercase name on any platform.
#
# This file is not the authoritative source of truth; FindMattes.py is.
# It is treated as an unchanged helper from the transcript and
# must not be rewritten by the Rotoscope Studio agent unless a bug fix
# is strictly necessary.
from .FindMattes import (
    fcn,
    getRotoModel,
    decode_segmap,
    createMatte,
    createMatteBatch,
)

__all__ = ['fcn', 'getRotoModel', 'decode_segmap', 'createMatte', 'createMatteBatch']
