""" python-libtiepie - Python interface for libtiepie-hw library

Copyright (c) 2025 TiePie engineering

Website: https://www.tiepie.com/LibTiePie

"""

from .api import api
from .const import *
from .library import library


class OscilloscopeTrigger(object):
    """"""

    def __init__(self, handle):
        self._handle = handle

    def _get_timeout(self):
        """  """
        value = api.tiepie_hw_oscilloscope_trigger_get_timeout(self._handle)
        library.check_last_status_raise_on_error()
        return value

    def _set_timeout(self, value):
        api.tiepie_hw_oscilloscope_trigger_set_timeout(self._handle, value)
        library.check_last_status_raise_on_error()

    def verify_timeout(self, value):
        """ Verify if a required trigger time out can be set, without actually setting the hardware itself.

        :param value: The required trigger time out in seconds, or #TIEPIE_HW_TO_INFINITY.
        :returns: The trigger time out that would have been set, if tiepie_hw_oscilloscope_trigger_set_timeout() was used.
        .. version added:: 1.0
        """
        result = api.tiepie_hw_oscilloscope_trigger_verify_timeout(self._handle, value)
        library.check_last_status_raise_on_error()
        return result

    def verify_timeout_ex(self, value, measure_mode, sample_rate):
        """ Verify if a required trigger time out can be set, without actually setting the hardware itself.

        :param value: The required trigger time out in seconds, or #TIEPIE_HW_TO_INFINITY.
        :param measure_mode: Measure mode, a TIEPIE_HW_MM_* value.
        :param sample_rate: Sample rate in Hz.
        :returns: The trigger time out that would have been set, if tiepie_hw_oscilloscope_trigger_set_timeout() was used.
        .. version added:: 1.0
        """
        result = api.tiepie_hw_oscilloscope_trigger_verify_timeout_ex(self._handle, value, measure_mode, sample_rate)
        library.check_last_status_raise_on_error()
        return result

    def _get_has_delay(self):
        """  """
        value = api.tiepie_hw_oscilloscope_trigger_has_delay(self._handle)
        library.check_last_status_raise_on_error()
        return value != BOOL_FALSE

    def has_delay_ex(self, measure_mode):
        """ Check whether the oscilloscope has trigger delay support for a specified measure mode.

        :param measure_mode: Measure mode, a TIEPIE_HW_MM_* value.
        :returns: ``True`` if the oscilloscope has trigger delay support, ``False`` otherwise.
        .. version added:: 1.0
        """
        result = api.tiepie_hw_oscilloscope_trigger_has_delay_ex(self._handle, measure_mode)
        library.check_last_status_raise_on_error()
        return result != BOOL_FALSE

    def _get_delay_max(self):
        """ Maximum trigger delay in seconds, for the currently selected measure mode and sample rate. """
        value = api.tiepie_hw_oscilloscope_trigger_get_delay_max(self._handle)
        library.check_last_status_raise_on_error()
        return value

    def get_delay_max_ex(self, measure_mode, sample_rate):
        """ Get the maximum trigger delay in seconds, for a specified measure mode and sample rate.

        :param measure_mode: Measure mode, a TIEPIE_HW_MM_* value.
        :param sample_rate: Sample rate in Hz.
        :returns: The maximum trigger delay in seconds.
        .. version added:: 1.0
        """
        result = api.tiepie_hw_oscilloscope_trigger_get_delay_max_ex(self._handle, measure_mode, sample_rate)
        library.check_last_status_raise_on_error()
        return result

    def _get_delay(self):
        """ Currently selected trigger delay in seconds. """
        value = api.tiepie_hw_oscilloscope_trigger_get_delay(self._handle)
        library.check_last_status_raise_on_error()
        return value

    def _set_delay(self, value):
        api.tiepie_hw_oscilloscope_trigger_set_delay(self._handle, value)
        library.check_last_status_raise_on_error()

    def verify_delay(self, value):
        """ Verify if a required trigger delay can be set, without actually setting the hardware itself.

        :param value: The required trigger delay in seconds.
        :returns: The trigger delay that would have been set, if tiepie_hw_oscilloscope_trigger_set_delay() was used.
        .. version added:: 1.0
        """
        result = api.tiepie_hw_oscilloscope_trigger_verify_delay(self._handle, value)
        library.check_last_status_raise_on_error()
        return result

    def verify_delay_ex(self, value, measure_mode, sample_rate):
        """ Verify if a required trigger delay can be set, without actually setting the hardware itself.

        :param value: The required trigger delay in seconds.
        :param measure_mode: Measure mode, a TIEPIE_HW_MM_* value.
        :param sample_rate: Sample rate in Hz.
        :returns: The trigger delay that would have been set, if tiepie_hw_oscilloscope_trigger_set_delay() was used.
        .. version added:: 1.0
        """
        result = api.tiepie_hw_oscilloscope_trigger_verify_delay_ex(self._handle, value, measure_mode, sample_rate)
        library.check_last_status_raise_on_error()
        return result

    timeout = property(_get_timeout, _set_timeout)
    has_delay = property(_get_has_delay)
    delay_max = property(_get_delay_max)
    delay = property(_get_delay, _set_delay)
