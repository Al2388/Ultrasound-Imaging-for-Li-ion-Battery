""" python-libtiepie - Python interface for libtiepie-hw library

Copyright (c) 2025 TiePie engineering

Website: https://www.tiepie.com/LibTiePie

"""

from .objectlist import ObjectList
from .api import api
from .library import library
from .triggeroutput import TriggerOutput


class TriggerOutputs(ObjectList):
    """"""

    def __init__(self, handle):
        super(TriggerOutputs, self).__init__()
        self._handle = handle
        self._items = [TriggerOutput(handle, i)
                       for i in range(api.tiepie_hw_device_trigger_get_output_count(handle))]

    def get_by_id(self, id):
        index = api.tiepie_hw_device_trigger_get_output_index_by_id(self._handle, id)
        library.check_last_status_raise_on_error()
        if index < len(self._items):
            return self._items[index]
        return None
