""" python-libtiepie - Python interface for libtiepie-hw library

Copyright (c) 2025 TiePie engineering

Website: https://www.tiepie.com/LibTiePie

"""

from .api import api
from .const import *
from .library import library
from .exceptions import *


class OscilloscopeChannelTriggerHystereses(object):
    """"""

    def __init__(self, handle, ch):
        self._handle = handle
        self._ch = ch

    def __getitem__(self, index):
        try:
            value = api.tiepie_hw_oscilloscope_channel_trigger_get_hysteresis(self._handle, self._ch, index)
            library.check_last_status_raise_on_error()
            return value
        except InvalidIndexError:
            raise IndexError('Index out of range')

    def __setitem__(self, index, value):
        try:
            api.tiepie_hw_oscilloscope_channel_trigger_set_hysteresis(self._handle, self._ch, index, value)
            library.check_last_status_raise_on_error()
        except InvalidIndexError:
            raise IndexError('Index out of range')

    def __len__(self):
        return self.count

    def _get_count(self):
        return api.tiepie_hw_oscilloscope_channel_trigger_get_hysteresis_count(self._handle, self._ch)

    count = property(_get_count)
