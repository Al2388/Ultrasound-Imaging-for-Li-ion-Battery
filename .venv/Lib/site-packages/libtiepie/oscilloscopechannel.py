""" python-libtiepie - Python interface for libtiepie-hw library

Copyright (c) 2025 TiePie engineering

Website: https://www.tiepie.com/LibTiePie

"""

from array import array
from ctypes import *
from .api import api
from .const import *
from .library import library
from .oscilloscopechanneltrigger import OscilloscopeChannelTrigger


class OscilloscopeChannel(object):
    """"""

    def __init__(self, handle, ch):
        self._handle = handle
        self._ch = ch
        self._trigger = OscilloscopeChannelTrigger(handle, ch) if self.has_trigger else None

    def _get_trigger(self):
        return self._trigger

    def _get_is_available(self):
        """ Check whether the channel is available. """
        value = api.tiepie_hw_oscilloscope_channel_is_available(self._handle, self._ch)
        library.check_last_status_raise_on_error()
        return value != BOOL_FALSE

    def is_available_ex(self, measure_mode, sample_rate, resolution, channel_enabled, channel_count):
        """ Check whether the channel is available, for a specific configuration.

        :param measure_mode: Measure mode, a TIEPIE_HW_MM_* value.
        :param sample_rate: Sample rate in Hz.
        :param resolution: Resolution in bits.
        :param channel_enabled: Pointer to buffer with channel enables.
        :param channel_count: Number of items in ``channel_enabled``
        :returns: ``True`` if available, ``False`` otherwise.
        .. version added:: 1.0
        """
        result = api.tiepie_hw_oscilloscope_channel_is_available_ex(self._handle, self._ch, measure_mode, sample_rate, resolution, channel_enabled, channel_count)
        library.check_last_status_raise_on_error()
        return result != BOOL_FALSE

    def _get_connector_type(self):
        """  """
        value = api.tiepie_hw_oscilloscope_channel_get_connector_type(self._handle, self._ch)
        library.check_last_status_raise_on_error()
        return value

    def _get_is_differential(self):
        """ Check whether the channel has a differential input. """
        value = api.tiepie_hw_oscilloscope_channel_is_differential(self._handle, self._ch)
        library.check_last_status_raise_on_error()
        return value != BOOL_FALSE

    def _get_is_isolated(self):
        """ Check whether the channel has a galvanically isolated input. """
        value = api.tiepie_hw_oscilloscope_channel_is_isolated(self._handle, self._ch)
        library.check_last_status_raise_on_error()
        return value != BOOL_FALSE

    def _get_impedance(self):
        """ Channel input impedance. """
        value = api.tiepie_hw_oscilloscope_channel_get_impedance(self._handle, self._ch)
        library.check_last_status_raise_on_error()
        return value

    def _get_bandwidths(self):
        """  """
        count = api.tiepie_hw_oscilloscope_channel_get_bandwidths(self._handle, self._ch, None, 0)
        library.check_last_status_raise_on_error()
        values = (c_double * count)()
        api.tiepie_hw_oscilloscope_channel_get_bandwidths(self._handle, self._ch, values, count)
        library.check_last_status_raise_on_error()
        return array('d', values)

    def _get_bandwidth(self):
        """ Current channel input bandwidth. """
        value = api.tiepie_hw_oscilloscope_channel_get_bandwidth(self._handle, self._ch)
        library.check_last_status_raise_on_error()
        return value

    def _set_bandwidth(self, value):
        api.tiepie_hw_oscilloscope_channel_set_bandwidth(self._handle, self._ch, value)
        library.check_last_status_raise_on_error()

    def _get_couplings(self):
        """  """
        value = api.tiepie_hw_oscilloscope_channel_get_couplings(self._handle, self._ch)
        library.check_last_status_raise_on_error()
        return value

    def _get_coupling(self):
        """ Currently set coupling. """
        value = api.tiepie_hw_oscilloscope_channel_get_coupling(self._handle, self._ch)
        library.check_last_status_raise_on_error()
        return value

    def _set_coupling(self, value):
        api.tiepie_hw_oscilloscope_channel_set_coupling(self._handle, self._ch, value)
        library.check_last_status_raise_on_error()

    def _get_enabled(self):
        """  """
        value = api.tiepie_hw_oscilloscope_channel_get_enabled(self._handle, self._ch)
        library.check_last_status_raise_on_error()
        return value != BOOL_FALSE

    def _set_enabled(self, value):
        value = BOOL_TRUE if value else BOOL_FALSE
        api.tiepie_hw_oscilloscope_channel_set_enabled(self._handle, self._ch, value)
        library.check_last_status_raise_on_error()

    def _get_auto_ranging(self):
        """  """
        value = api.tiepie_hw_oscilloscope_channel_get_auto_ranging(self._handle, self._ch)
        library.check_last_status_raise_on_error()
        return value != BOOL_FALSE

    def _set_auto_ranging(self, value):
        value = BOOL_TRUE if value else BOOL_FALSE
        api.tiepie_hw_oscilloscope_channel_set_auto_ranging(self._handle, self._ch, value)
        library.check_last_status_raise_on_error()

    def _get_ranges(self):
        """ Supported input ranges, with the currently selected coupling. """
        count = api.tiepie_hw_oscilloscope_channel_get_ranges(self._handle, self._ch, None, 0)
        library.check_last_status_raise_on_error()
        values = (c_double * count)()
        api.tiepie_hw_oscilloscope_channel_get_ranges(self._handle, self._ch, values, count)
        library.check_last_status_raise_on_error()
        return array('d', values)

    def get_ranges_ex(self, coupling, list, length):
        """ Get the supported ranges by coupling.

        :param coupling: Coupling: a TIEPIE_HW_CK_* value.
        :param list: Pointer to array.
        :param length: Number of elements in array.
        :returns: Total number of ranges.
        .. version added:: 1.0
        """
        result = api.tiepie_hw_oscilloscope_channel_get_ranges_ex(self._handle, self._ch, coupling, list, length)
        library.check_last_status_raise_on_error()
        return result

    def _get_range(self):
        """ Currently selected input range. """
        value = api.tiepie_hw_oscilloscope_channel_get_range(self._handle, self._ch)
        library.check_last_status_raise_on_error()
        return value

    def _set_range(self, value):
        api.tiepie_hw_oscilloscope_channel_set_range(self._handle, self._ch, value)
        library.check_last_status_raise_on_error()

    def _get_has_safeground(self):
        """  """
        value = api.tiepie_hw_oscilloscope_channel_has_safeground(self._handle, self._ch)
        library.check_last_status_raise_on_error()
        return value != BOOL_FALSE

    def _get_safeground_enabled(self):
        """ Check whether SafeGround is enabled. """
        value = api.tiepie_hw_oscilloscope_channel_get_safeground_enabled(self._handle, self._ch)
        library.check_last_status_raise_on_error()
        return value != BOOL_FALSE

    def _set_safeground_enabled(self, value):
        value = BOOL_TRUE if value else BOOL_FALSE
        api.tiepie_hw_oscilloscope_channel_set_safeground_enabled(self._handle, self._ch, value)
        library.check_last_status_raise_on_error()

    def _get_safeground_threshold_min(self):
        """ Minimum SafeGround threshold current. """
        value = api.tiepie_hw_oscilloscope_channel_get_safeground_threshold_min(self._handle, self._ch)
        library.check_last_status_raise_on_error()
        return value

    def _get_safeground_threshold_max(self):
        """ Maximum SafeGround threshold current. """
        value = api.tiepie_hw_oscilloscope_channel_get_safeground_threshold_max(self._handle, self._ch)
        library.check_last_status_raise_on_error()
        return value

    def _get_safeground_threshold(self):
        """ Actual SafeGround threshold current. """
        value = api.tiepie_hw_oscilloscope_channel_get_safeground_threshold(self._handle, self._ch)
        library.check_last_status_raise_on_error()
        return value

    def _set_safeground_threshold(self, value):
        api.tiepie_hw_oscilloscope_channel_set_safeground_threshold(self._handle, self._ch, value)
        library.check_last_status_raise_on_error()

    def verify_safeground_threshold(self, threshold):
        """ Verify if the required threshold current can be set.

        :param threshold: The required threshold current, in Ampere.
        :returns: The SafeGround threshold current that would be set.
        .. version added:: 1.0
        """
        result = api.tiepie_hw_oscilloscope_channel_verify_safeground_threshold(self._handle, self._ch, threshold)
        library.check_last_status_raise_on_error()
        return result

    def _get_has_trigger(self):
        """  """
        value = api.tiepie_hw_oscilloscope_channel_has_trigger(self._handle, self._ch)
        library.check_last_status_raise_on_error()
        return value != BOOL_FALSE

    def has_trigger_ex(self, measure_mode):
        """ Check whether the specified channel has trigger support, for a specific configuration.

        :param measure_mode: Measure mode, a TIEPIE_HW_MM_* value.
        :returns: ``True`` if the channel has trigger support, ``False`` otherwise.
        .. version added:: 1.0
        """
        result = api.tiepie_hw_oscilloscope_channel_has_trigger_ex(self._handle, self._ch, measure_mode)
        library.check_last_status_raise_on_error()
        return result != BOOL_FALSE

    def _get_data_value_min(self):
        """ Minimum value of the input range the current data was measured with. """
        value = api.tiepie_hw_oscilloscope_channel_get_data_value_min(self._handle, self._ch)
        library.check_last_status_raise_on_error()
        return value

    def _get_data_value_max(self):
        """ Maximum value of the input range the current data was measured with. """
        value = api.tiepie_hw_oscilloscope_channel_get_data_value_max(self._handle, self._ch)
        library.check_last_status_raise_on_error()
        return value

    def _get_data_raw_type(self):
        """ Get raw data type. """
        value = api.tiepie_hw_oscilloscope_channel_get_data_raw_type(self._handle, self._ch)
        library.check_last_status_raise_on_error()
        return value

    def _get_data_raw_value_min(self):
        """ Get possible raw data minimum value. """
        value = api.tiepie_hw_oscilloscope_channel_get_data_raw_value_min(self._handle, self._ch)
        library.check_last_status_raise_on_error()
        return value

    def _get_data_raw_value_zero(self):
        """ Get raw data value which equals zero. """
        value = api.tiepie_hw_oscilloscope_channel_get_data_raw_value_zero(self._handle, self._ch)
        library.check_last_status_raise_on_error()
        return value

    def _get_data_raw_value_max(self):
        """ Get possible raw data maximum value. """
        value = api.tiepie_hw_oscilloscope_channel_get_data_raw_value_max(self._handle, self._ch)
        library.check_last_status_raise_on_error()
        return value

    def _get_is_range_max_reachable(self):
        """ Check whether the ranges maximum is reachable. """
        value = api.tiepie_hw_oscilloscope_channel_is_range_max_reachable(self._handle, self._ch)
        library.check_last_status_raise_on_error()
        return value != BOOL_FALSE

    def _get_has_sureconnect(self):
        """ Check whether a specified channel supports SureConnect connection testing. """
        value = api.tiepie_hw_oscilloscope_channel_has_sureconnect(self._handle, self._ch)
        library.check_last_status_raise_on_error()
        return value != BOOL_FALSE

    is_available = property(_get_is_available)
    connector_type = property(_get_connector_type)
    is_differential = property(_get_is_differential)
    is_isolated = property(_get_is_isolated)
    impedance = property(_get_impedance)
    bandwidths = property(_get_bandwidths)
    bandwidth = property(_get_bandwidth, _set_bandwidth)
    couplings = property(_get_couplings)
    coupling = property(_get_coupling, _set_coupling)
    enabled = property(_get_enabled, _set_enabled)
    auto_ranging = property(_get_auto_ranging, _set_auto_ranging)
    ranges = property(_get_ranges)
    range = property(_get_range, _set_range)
    has_safeground = property(_get_has_safeground)
    safeground_enabled = property(_get_safeground_enabled, _set_safeground_enabled)
    safeground_threshold_min = property(_get_safeground_threshold_min)
    safeground_threshold_max = property(_get_safeground_threshold_max)
    safeground_threshold = property(_get_safeground_threshold, _set_safeground_threshold)
    has_trigger = property(_get_has_trigger)
    data_value_min = property(_get_data_value_min)
    data_value_max = property(_get_data_value_max)
    data_raw_type = property(_get_data_raw_type)
    data_raw_value_min = property(_get_data_raw_value_min)
    data_raw_value_zero = property(_get_data_raw_value_zero)
    data_raw_value_max = property(_get_data_raw_value_max)
    is_range_max_reachable = property(_get_is_range_max_reachable)
    has_sureconnect = property(_get_has_sureconnect)
    trigger = property(_get_trigger)
