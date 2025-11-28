""" python-libtiepie - Python interface for libtiepie-hw library

Copyright (c) 2025 TiePie engineering

Website: https://www.tiepie.com/LibTiePie

"""

from .api import api
from .const import *
from .library import library
from .oscilloscopechanneltriggerlevels import OscilloscopeChannelTriggerLevels
from .oscilloscopechanneltriggerhystereses import OscilloscopeChannelTriggerHystereses
from .oscilloscopechanneltriggertimes import OscilloscopeChannelTriggerTimes


class OscilloscopeChannelTrigger(object):
    """"""

    def __init__(self, handle, ch):
        self._handle = handle
        self._ch = ch
        self._levels = OscilloscopeChannelTriggerLevels(handle, ch)
        self._hystereses = OscilloscopeChannelTriggerHystereses(handle, ch)
        self._times = OscilloscopeChannelTriggerTimes(handle, ch)

    def _get_levels(self):
        return self._levels

    def _get_hystereses(self):
        return self._hystereses

    def _get_times(self):
        return self._times

    def _get_is_available(self):
        """ Check whether the channel trigger is available, with the current oscilloscope settings. """
        value = api.tiepie_hw_oscilloscope_channel_trigger_is_available(self._handle, self._ch)
        library.check_last_status_raise_on_error()
        return value != BOOL_FALSE

    def is_available_ex(self, measure_mode, sample_rate, resolution, channel_enabled, channel_trigger_enabled, channel_count):
        """ Check whether the channel trigger is available, for a specific configuration.

        :param measure_mode: Measure mode, a TIEPIE_HW_MM_* value.
        :param sample_rate: sample rate in Hz.
        :param resolution: Resolution in bits.
        :param channel_enabled: Pointer to buffer with channel enables.
        :param channel_trigger_enabled: Pointer to buffer with channel trigger enables.
        :param channel_count: Number of items in ``channel_enabled`` and ``channel_trigger_enabled.``
        :returns: ``True`` if available, ``False`` otherwise.
        .. version added:: 1.0
        """
        result = api.tiepie_hw_oscilloscope_channel_trigger_is_available_ex(self._handle, self._ch, measure_mode, sample_rate, resolution, channel_enabled, channel_trigger_enabled, channel_count)
        library.check_last_status_raise_on_error()
        return result != BOOL_FALSE

    def _get_is_triggered(self):
        """ Check whether the channel trigger caused a trigger. """
        value = api.tiepie_hw_oscilloscope_channel_trigger_is_triggered(self._handle, self._ch)
        library.check_last_status_raise_on_error()
        return value != BOOL_FALSE

    def _get_enabled(self):
        """  """
        value = api.tiepie_hw_oscilloscope_channel_trigger_get_enabled(self._handle, self._ch)
        library.check_last_status_raise_on_error()
        return value != BOOL_FALSE

    def _set_enabled(self, value):
        value = BOOL_TRUE if value else BOOL_FALSE
        api.tiepie_hw_oscilloscope_channel_trigger_set_enabled(self._handle, self._ch, value)
        library.check_last_status_raise_on_error()

    def _get_kinds(self):
        """  """
        value = api.tiepie_hw_oscilloscope_channel_trigger_get_kinds(self._handle, self._ch)
        library.check_last_status_raise_on_error()
        return value

    def get_kinds_ex(self, measure_mode):
        """ Get the supported channel trigger kinds, for a specific measure mode.

        :param measure_mode: Measure mode, a TIEPIE_HW_MM_* value.
        :returns: Supported trigger kinds, a set of OR-ed TIEPIE_HW_TK_* values.
        .. version added:: 1.0
        """
        result = api.tiepie_hw_oscilloscope_channel_trigger_get_kinds_ex(self._handle, self._ch, measure_mode)
        library.check_last_status_raise_on_error()
        return result

    def _get_kind(self):
        """ Currently selected channel trigger kind. """
        value = api.tiepie_hw_oscilloscope_channel_trigger_get_kind(self._handle, self._ch)
        library.check_last_status_raise_on_error()
        return value

    def _set_kind(self, value):
        api.tiepie_hw_oscilloscope_channel_trigger_set_kind(self._handle, self._ch, value)
        library.check_last_status_raise_on_error()

    def _get_level_modes(self):
        """  """
        value = api.tiepie_hw_oscilloscope_channel_trigger_get_level_modes(self._handle, self._ch)
        library.check_last_status_raise_on_error()
        return value

    def _get_level_mode(self):
        """ Current trigger level mode. """
        value = api.tiepie_hw_oscilloscope_channel_trigger_get_level_mode(self._handle, self._ch)
        library.check_last_status_raise_on_error()
        return value

    def _set_level_mode(self, value):
        api.tiepie_hw_oscilloscope_channel_trigger_set_level_mode(self._handle, self._ch, value)
        library.check_last_status_raise_on_error()

    def _get_conditions(self):
        """  """
        value = api.tiepie_hw_oscilloscope_channel_trigger_get_conditions(self._handle, self._ch)
        library.check_last_status_raise_on_error()
        return value

    def get_conditions_ex(self, measure_mode, trigger_kind):
        """ Get the supported trigger conditions, for a specific measure mode and trigger kind.

        :param measure_mode: Measure mode, a TIEPIE_HW_MM_* value.
        :param trigger_kind: Trigger kind, a TIEPIE_HW_TK_* value.
        :returns: Supported trigger conditions for this channel, measure mode and trigger kind, a set of OR-ed TIEPIE_HW_TC_* values.
        .. version added:: 1.0
        """
        result = api.tiepie_hw_oscilloscope_channel_trigger_get_conditions_ex(self._handle, self._ch, measure_mode, trigger_kind)
        library.check_last_status_raise_on_error()
        return result

    def _get_condition(self):
        """ Current selected trigger condition. """
        value = api.tiepie_hw_oscilloscope_channel_trigger_get_condition(self._handle, self._ch)
        library.check_last_status_raise_on_error()
        return value

    def _set_condition(self, value):
        api.tiepie_hw_oscilloscope_channel_trigger_set_condition(self._handle, self._ch, value)
        library.check_last_status_raise_on_error()

    def verify_time_ex(self, index, value, measure_mode, sample_rate, trigger_kind, trigger_condition):
        """ Verify if the required trigger time value, measure mode, sample rate, trigger type and trigger condition can be set.

        :param index: The trigger time index, ``0`` to <tt>tiepie_hw_oscilloscope_channel_trigger_get_time_count() - 1</tt>.
        :param value: The required trigger time value, in seconds.
        :param measure_mode: The required measure mode.
        :param sample_rate: Sample rate in Hz.
        :param trigger_kind: The required trigger kind.
        :param trigger_condition: The required trigger condition.
        :returns: The actually trigger time value that would have been set, in seconds.
        .. version added:: 1.0
        """
        result = api.tiepie_hw_oscilloscope_channel_trigger_verify_time_ex(self._handle, self._ch, index, value, measure_mode, sample_rate, trigger_kind, trigger_condition)
        library.check_last_status_raise_on_error()
        return result

    is_available = property(_get_is_available)
    is_triggered = property(_get_is_triggered)
    enabled = property(_get_enabled, _set_enabled)
    kinds = property(_get_kinds)
    kind = property(_get_kind, _set_kind)
    level_modes = property(_get_level_modes)
    level_mode = property(_get_level_mode, _set_level_mode)
    conditions = property(_get_conditions)
    condition = property(_get_condition, _set_condition)
    levels = property(_get_levels)
    hystereses = property(_get_hystereses)
    times = property(_get_times)
