""" python-libtiepie - Python interface for libtiepie-hw library

Copyright (c) 2025 TiePie engineering

Website: https://www.tiepie.com/LibTiePie

"""

from ctypes import *
from .api import api
from .const import *
from .library import library


class TriggerInput(object):
    """"""

    def __init__(self, handle, index):
        self._handle = handle
        self._index = index

    def _get_is_triggered(self):
        """ Check whether the trigger input caused a trigger. """
        value = api.tiepie_hw_oscilloscope_trigger_input_is_triggered(self._handle, self._index)
        library.check_last_status_raise_on_error()
        return value != BOOL_FALSE

    def _get_enabled(self):
        """  """
        value = api.tiepie_hw_device_trigger_input_get_enabled(self._handle, self._index)
        library.check_last_status_raise_on_error()
        return value != BOOL_FALSE

    def _set_enabled(self, value):
        value = BOOL_TRUE if value else BOOL_FALSE
        api.tiepie_hw_device_trigger_input_set_enabled(self._handle, self._index, value)
        library.check_last_status_raise_on_error()

    def _get_kinds(self):
        """  """
        value = api.tiepie_hw_device_trigger_input_get_kinds(self._handle, self._index)
        library.check_last_status_raise_on_error()
        return value

    def get_kinds_ex(self, measure_mode):
        """ Get the supported trigger kinds trigger input and measure mode.

        :param measure_mode: Measure mode, a TIEPIE_HW_MM_* value.
        :returns: Supported trigger input kinds, a set of OR-ed TIEPIE_HW_TK_* values.
        .. version added:: 1.0
        """
        result = api.tiepie_hw_oscilloscope_trigger_input_get_kinds_ex(self._handle, self._index, measure_mode)
        library.check_last_status_raise_on_error()
        return result

    def _get_kind(self):
        """ Currently selected trigger kind. """
        value = api.tiepie_hw_device_trigger_input_get_kind(self._handle, self._index)
        library.check_last_status_raise_on_error()
        return value

    def _set_kind(self, value):
        api.tiepie_hw_device_trigger_input_set_kind(self._handle, self._index, value)
        library.check_last_status_raise_on_error()

    def _get_is_available(self):
        """  """
        value = api.tiepie_hw_device_trigger_input_is_available(self._handle, self._index)
        library.check_last_status_raise_on_error()
        return value != BOOL_FALSE

    def is_available_ex(self, measure_mode):
        """ Check whether a device trigger input is available, for a specific measure mode.

        :param measure_mode: Measure mode, a TIEPIE_HW_MM_* value.
        :returns: ``True`` if available, ``False`` otherwise.
        .. version added:: 1.0
        """
        result = api.tiepie_hw_oscilloscope_trigger_input_is_available_ex(self._handle, self._index, measure_mode)
        library.check_last_status_raise_on_error()
        return result != BOOL_FALSE

    def _get_id(self):
        """ Id. """
        value = api.tiepie_hw_device_trigger_input_get_id(self._handle, self._index)
        library.check_last_status_raise_on_error()
        return value

    def _get_name(self):
        """ Name. """
        length = api.tiepie_hw_device_trigger_input_get_name(self._handle, self._index, None, 0)
        library.check_last_status_raise_on_error()
        buf = create_string_buffer(length + 1)
        api.tiepie_hw_device_trigger_input_get_name(self._handle, self._index, buf, length + 1)
        library.check_last_status_raise_on_error()
        return buf.value.decode('utf-8')

    is_triggered = property(_get_is_triggered)
    enabled = property(_get_enabled, _set_enabled)
    kinds = property(_get_kinds)
    kind = property(_get_kind, _set_kind)
    is_available = property(_get_is_available)
    id = property(_get_id)
    name = property(_get_name)
