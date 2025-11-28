""" python-libtiepie - Python interface for libtiepie-hw library

Copyright (c) 2025 TiePie engineering

Website: https://www.tiepie.com/LibTiePie

"""

from ctypes import *
from .api import api
from .const import *
from .library import library


class TriggerOutput(object):
    """"""

    def __init__(self, handle, index):
        self._handle = handle
        self._index = index

    def _get_enabled(self):
        """  """
        value = api.tiepie_hw_device_trigger_output_get_enabled(self._handle, self._index)
        library.check_last_status_raise_on_error()
        return value != BOOL_FALSE

    def _set_enabled(self, value):
        value = BOOL_TRUE if value else BOOL_FALSE
        api.tiepie_hw_device_trigger_output_set_enabled(self._handle, self._index, value)
        library.check_last_status_raise_on_error()

    def _get_events(self):
        """  """
        value = api.tiepie_hw_device_trigger_output_get_events(self._handle, self._index)
        library.check_last_status_raise_on_error()
        return value

    def _get_event(self):
        """ Currently selected trigger output event. """
        value = api.tiepie_hw_device_trigger_output_get_event(self._handle, self._index)
        library.check_last_status_raise_on_error()
        return value

    def _set_event(self, value):
        api.tiepie_hw_device_trigger_output_set_event(self._handle, self._index, value)
        library.check_last_status_raise_on_error()

    def _get_id(self):
        """  """
        value = api.tiepie_hw_device_trigger_output_get_id(self._handle, self._index)
        library.check_last_status_raise_on_error()
        return value

    def _get_name(self):
        """ Name. """
        length = api.tiepie_hw_device_trigger_output_get_name(self._handle, self._index, None, 0)
        library.check_last_status_raise_on_error()
        buf = create_string_buffer(length + 1)
        api.tiepie_hw_device_trigger_output_get_name(self._handle, self._index, buf, length + 1)
        library.check_last_status_raise_on_error()
        return buf.value.decode('utf-8')

    def trigger(self):
        """ Trigger the specified device trigger output.

        :returns: ``True`` if successful, ``False`` otherwise.
        .. version added:: 1.0
        """
        result = api.tiepie_hw_device_trigger_output_trigger(self._handle, self._index)
        library.check_last_status_raise_on_error()
        return result != BOOL_FALSE

    enabled = property(_get_enabled, _set_enabled)
    events = property(_get_events)
    event = property(_get_event, _set_event)
    id = property(_get_id)
    name = property(_get_name)
