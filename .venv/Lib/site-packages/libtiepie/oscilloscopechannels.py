""" python-libtiepie - Python interface for libtiepie-hw library

Copyright (c) 2025 TiePie engineering

Website: https://www.tiepie.com/LibTiePie

"""

from .objectlist import ObjectList
from .api import api
from .oscilloscopechannel import OscilloscopeChannel


class OscilloscopeChannels(ObjectList):
    """"""

    def __init__(self, handle):
        super(OscilloscopeChannels, self).__init__()
        self._items = [OscilloscopeChannel(handle, i)
                       for i in range(api.tiepie_hw_oscilloscope_get_channel_count(handle))]
