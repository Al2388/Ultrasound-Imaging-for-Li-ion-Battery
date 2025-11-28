""" python-libtiepie - Python interface for libtiepie-hw library

Copyright (c) 2025 TiePie engineering

Website: https://www.tiepie.com/LibTiePie

"""

from array import array
import numpy as np
from ctypes import c_uint8, c_float
from .api import api
from .const import *
from .utils import *
from .library import library
from .device import Device
from .oscilloscopechannels import OscilloscopeChannels
from .exceptions import *
from .oscilloscopetrigger import OscilloscopeTrigger


class Oscilloscope(Device):
    """"""

    def __init__(self, handle):
        super(Oscilloscope, self).__init__(handle)
        self._channels = OscilloscopeChannels(handle)
        self._trigger = OscilloscopeTrigger(handle)

    def _get_channels(self):
        return self._channels

    def _get_trigger(self):
        return self._trigger

    def _get_is_demo(self):
        """  """
        value = api.tiepie_hw_oscilloscope_is_demo(self._handle)
        library.check_last_status_raise_on_error()
        return value != BOOL_FALSE

    def get_data(self, count=None, raw=False):
        """ Get the measurement data for enabled channels.

        :param count: Number of samples to read, defaults to all.
        :param raw: Get raw data.

        :returns: `list` of `array.array`'s with float sample data.
        .. version added:: 1.0
        """
        if not self.is_data_ready:
            raise UnsuccessfulError()

        channel_count = len(self.channels)

        # Calculate valid data start/length:
        if self._measure_mode == MM_BLOCK:
            length = int(self._record_length - round(self._pre_sample_ratio * self._record_length) + self.valid_pre_sample_count)
            start = self._record_length - length
        else:
            length = self._record_length
            start = 0

        if (count is not None) and (count >= 0) and (count < length):
            length = count

        # Create pointer array:
        pointers = [c_void_p(0)] * channel_count

        # Allocate memory and fill pointer array:
        result = [[]] * channel_count
        for i in range(channel_count):
            if self._active_channels[i]:
                if raw:
                    raw_type = self.channels[i].data_raw_type
                    if raw_type == DATARAWTYPE_INT8:
                        result[i] = array('b', [0]) * length
                    elif raw_type == DATARAWTYPE_INT16:
                        result[i] = array('h', [0]) * length
                    elif raw_type == DATARAWTYPE_INT32:
                        result[i] = array('l', [0]) * length
                    elif raw_type == DATARAWTYPE_INT64:
                        result[i] = array('q', [0]) * length
                    elif raw_type == DATARAWTYPE_UINT8:
                        result[i] = array('B', [0]) * length
                    elif raw_type == DATARAWTYPE_UINT16:
                        result[i] = array('H', [0]) * length
                    elif raw_type == DATARAWTYPE_UINT32:
                        result[i] = array('L', [0]) * length
                    elif raw_type == DATARAWTYPE_UINT64:
                        result[i] = array('Q', [0]) * length
                    elif raw_type == DATARAWTYPE_FLOAT32:
                        result[i] = array('f', [0]) * length
                    elif raw_type == DATARAWTYPE_FLOAT64:
                        result[i] = array('d', [0]) * length
                    else:
                        raise UnsuccessfulError()
                else:
                    result[i] = array('f', [0]) * length
                pointers[i] = cast(result[i].buffer_info()[0], c_void_p)

        pointers = (c_void_p * len(pointers))(*pointers)

        # Get the data:
        if raw:
            api.tiepie_hw_oscilloscope_get_data_raw(self._handle, pointers, channel_count, start, length)
        else:
            api.tiepie_hw_oscilloscope_get_data(self._handle, pointers, channel_count, start, length)
        library.check_last_status_raise_on_error()

        return result

    def get_data_numpy(self, count=None):
        """ Get the measurement data for enabled channels.

        :param count: Number of samples to read, defaults to all.

        :returns: A 2-dimensional array with float sample data.
        .. version added:: 1.0
        """

        if not self.is_data_ready:
            raise UnsuccessfulError()

        channel_count = len(self.channels)

        # Calculate valid data start/length:
        if self._measure_mode == MM_BLOCK:
            length = int(self._record_length - round(self._pre_sample_ratio * self._record_length) + self.valid_pre_sample_count)
            start = self._record_length - length
        else:
            length = self._record_length
            start = 0

        if (count is not None) and (count >= 0) and (count < length):
            length = count

        result = np.empty([channel_count, length], dtype=np.float32)

        pointers = [c_void_p(0)] * channel_count
        for i in range(channel_count):
            if not self._active_channels[i]:
                result[i].fill(np.NAN)

            if self._active_channels[i]:
                pointers[i] = cast(np.ctypeslib.as_ctypes(result)[i], c_void_p)

        pointers = (c_void_p * len(pointers))(*pointers)

        api.tiepie_hw_oscilloscope_get_data(self._handle, pointers, channel_count, start, length)
        library.check_last_status_raise_on_error()

        return result

    def _get_valid_pre_sample_count(self):
        """ Number of valid pre samples in the measurement. """
        value = api.tiepie_hw_oscilloscope_get_valid_pre_sample_count(self._handle)
        library.check_last_status_raise_on_error()
        return value

    def get_data_raw(self, buffers, channel_count, start_index, sample_count):
        """ Get raw measurement data.

        :param buffers: Pointer to buffer with pointers to buffer for channel data, pointer buffer may contain ``None`` pointers.
        :param channel_count: Number of pointers in pointer buffer.
        :param start_index: Position in record to start reading.
        :param sample_count: Number of samples to read.
        :returns: Number of samples read.
        .. version added:: 1.0
        """
        result = api.tiepie_hw_oscilloscope_get_data_raw(self._handle, buffers, channel_count, start_index, sample_count)
        library.check_last_status_raise_on_error()
        return result

    def start(self):
        """  """
        if self.is_running:
            raise MeasurementRunningError()

        # Cache some values, needed for getting data:
        self._measure_mode = self.measure_mode
        self._record_length = self.record_length
        if self._measure_mode == MM_BLOCK:
            self._pre_sample_ratio = self.pre_sample_ratio
        self._active_channels = []
        for ch in self.channels:
            self._active_channels.append(ch.enabled)

        result = api.tiepie_hw_oscilloscope_start(self._handle)
        library.check_last_status_raise_on_error()
        return result

    def stop(self):
        """ Stop a running measurement.

        .. version added:: 1.0
        """
        api.tiepie_hw_oscilloscope_stop(self._handle)
        library.check_last_status_raise_on_error()

    def force_trigger(self):
        """ Force a trigger.

        .. version added:: 1.0
        """
        api.tiepie_hw_oscilloscope_force_trigger(self._handle)
        library.check_last_status_raise_on_error()

    def _get_measure_modes(self):
        """  """
        value = api.tiepie_hw_oscilloscope_get_measure_modes(self._handle)
        library.check_last_status_raise_on_error()
        return value

    def _get_measure_mode(self):
        """ Current measure mode. """
        value = api.tiepie_hw_oscilloscope_get_measure_mode(self._handle)
        library.check_last_status_raise_on_error()
        return value

    def _set_measure_mode(self, value):
        api.tiepie_hw_oscilloscope_set_measure_mode(self._handle, value)
        library.check_last_status_raise_on_error()

    def _get_is_running(self):
        """  """
        value = api.tiepie_hw_oscilloscope_is_running(self._handle)
        library.check_last_status_raise_on_error()
        return value != BOOL_FALSE

    def _get_is_triggered(self):
        """ Check whether the oscilloscope has triggered. """
        value = api.tiepie_hw_oscilloscope_is_triggered(self._handle)
        library.check_last_status_raise_on_error()
        return value != BOOL_FALSE

    def _get_is_timeout_triggered(self):
        """ Check whether the trigger was caused by the trigger time out. """
        value = api.tiepie_hw_oscilloscope_is_timeout_triggered(self._handle)
        library.check_last_status_raise_on_error()
        return value != BOOL_FALSE

    def _get_is_force_triggered(self):
        """ Check whether the trigger was caused by tiepie_hw_oscilloscope_force_trigger. """
        value = api.tiepie_hw_oscilloscope_is_force_triggered(self._handle)
        library.check_last_status_raise_on_error()
        return value != BOOL_FALSE

    def _get_is_data_ready(self):
        """ Check whether new, unread measured data is available. """
        value = api.tiepie_hw_oscilloscope_is_data_ready(self._handle)
        library.check_last_status_raise_on_error()
        return value != BOOL_FALSE

    def _get_is_data_overflow(self):
        """ Check whether a data overflow has occurred. """
        value = api.tiepie_hw_oscilloscope_is_data_overflow(self._handle)
        library.check_last_status_raise_on_error()
        return value != BOOL_FALSE

    def _get_auto_resolution_modes(self):
        """  """
        value = api.tiepie_hw_oscilloscope_get_auto_resolution_modes(self._handle)
        library.check_last_status_raise_on_error()
        return value

    def _get_auto_resolution_mode(self):
        """ Current auto resolution mode. """
        value = api.tiepie_hw_oscilloscope_get_auto_resolution_mode(self._handle)
        library.check_last_status_raise_on_error()
        return value

    def _set_auto_resolution_mode(self, value):
        api.tiepie_hw_oscilloscope_set_auto_resolution_mode(self._handle, value)
        library.check_last_status_raise_on_error()

    def _get_resolutions(self):
        """ :class:`array.array` of supported resolutions. """
        count = api.tiepie_hw_oscilloscope_get_resolutions(self._handle, None, 0)
        library.check_last_status_raise_on_error()
        values = (c_uint8 * count)()
        api.tiepie_hw_oscilloscope_get_resolutions(self._handle, values, count)
        library.check_last_status_raise_on_error()
        return array('B', values)

    def _get_resolution(self):
        """ Current resolution. """
        value = api.tiepie_hw_oscilloscope_get_resolution(self._handle)
        library.check_last_status_raise_on_error()
        return value

    def _set_resolution(self, value):
        api.tiepie_hw_oscilloscope_set_resolution(self._handle, value)
        library.check_last_status_raise_on_error()

    def _get_is_resolution_enhanced(self):
        """ Check whether the currently selected resolution is enhanced or a native resolution of the hardware. """
        value = api.tiepie_hw_oscilloscope_is_resolution_enhanced(self._handle)
        library.check_last_status_raise_on_error()
        return value != BOOL_FALSE

    def is_resolution_enhanced_ex(self, value):
        """ Check whether resolution is enhanced.

        :param value: Resolution in bits.
        :returns: ``True`` if resolution is enhanced, ``False`` otherwise.
        .. version added:: 1.0
        """
        result = api.tiepie_hw_oscilloscope_is_resolution_enhanced_ex(self._handle, value)
        library.check_last_status_raise_on_error()
        return result != BOOL_FALSE

    def _get_clock_sources(self):
        """  """
        value = api.tiepie_hw_oscilloscope_get_clock_sources(self._handle)
        library.check_last_status_raise_on_error()
        return value

    def _get_clock_source(self):
        """ Currently selected clock source. """
        value = api.tiepie_hw_oscilloscope_get_clock_source(self._handle)
        library.check_last_status_raise_on_error()
        return value

    def _set_clock_source(self, value):
        api.tiepie_hw_oscilloscope_set_clock_source(self._handle, value)
        library.check_last_status_raise_on_error()

    def _get_clock_source_frequencies(self):
        """ :class:`array.array` of supported clock source frequencies. """
        count = api.tiepie_hw_oscilloscope_get_clock_source_frequencies(self._handle, None, 0)
        library.check_last_status_raise_on_error()
        values = (c_double * count)()
        api.tiepie_hw_oscilloscope_get_clock_source_frequencies(self._handle, values, count)
        library.check_last_status_raise_on_error()
        return array('d', values)

    def get_clock_source_frequencies_ex(self, value, list, length):
        """ Get an array with the supported clock source frequencies for the specified clock source.

        :param value: The requested clock source, a TIEPIE_HW_CS_* value.
        :param list: A pointer to an array for the clock source frequencies, or ``None.``
        :param length: The number of elements in the array.
        :returns: Total number of supported clock source frequencies, or ``0`` when unsuccessful.
        .. version added:: 1.0
        """
        result = api.tiepie_hw_oscilloscope_get_clock_source_frequencies_ex(self._handle, value, list, length)
        library.check_last_status_raise_on_error()
        return result

    def _get_clock_source_frequency(self):
        """ Current clock source frequency. """
        value = api.tiepie_hw_oscilloscope_get_clock_source_frequency(self._handle)
        library.check_last_status_raise_on_error()
        return value

    def _set_clock_source_frequency(self, value):
        api.tiepie_hw_oscilloscope_set_clock_source_frequency(self._handle, value)
        library.check_last_status_raise_on_error()

    def _get_clock_outputs(self):
        """  """
        value = api.tiepie_hw_oscilloscope_get_clock_outputs(self._handle)
        library.check_last_status_raise_on_error()
        return value

    def _get_clock_output(self):
        """ Currently selected clock output. """
        value = api.tiepie_hw_oscilloscope_get_clock_output(self._handle)
        library.check_last_status_raise_on_error()
        return value

    def _set_clock_output(self, value):
        api.tiepie_hw_oscilloscope_set_clock_output(self._handle, value)
        library.check_last_status_raise_on_error()

    def _get_clock_output_frequencies(self):
        """ :class:`array.array` of supported clock output frequencies. """
        count = api.tiepie_hw_oscilloscope_get_clock_output_frequencies(self._handle, None, 0)
        library.check_last_status_raise_on_error()
        values = (c_double * count)()
        api.tiepie_hw_oscilloscope_get_clock_output_frequencies(self._handle, values, count)
        library.check_last_status_raise_on_error()
        return array('d', values)

    def get_clock_output_frequencies_ex(self, clock_output, list, length):
        """ Get an array with the supported clock output frequencies for the specified clock output.

        :param clock_output: The requested clock output, a TIEPIE_HW_CS_* value.
        :param list: A pointer to an array for the clock output frequencies, or ``None.``
        :param length: The number of elements in the array.
        :returns: Total number of supported clock output frequencies, or ``0`` when unsuccessful.
        .. version added:: 1.0
        """
        result = api.tiepie_hw_oscilloscope_get_clock_output_frequencies_ex(self._handle, clock_output, list, length)
        library.check_last_status_raise_on_error()
        return result

    def _get_clock_output_frequency(self):
        """ Current clock output frequency. """
        value = api.tiepie_hw_oscilloscope_get_clock_output_frequency(self._handle)
        library.check_last_status_raise_on_error()
        return value

    def _set_clock_output_frequency(self, value):
        api.tiepie_hw_oscilloscope_set_clock_output_frequency(self._handle, value)
        library.check_last_status_raise_on_error()

    def _get_sample_rate_max(self):
        """  """
        value = api.tiepie_hw_oscilloscope_get_sample_rate_max(self._handle)
        library.check_last_status_raise_on_error()
        return value

    def _get_sample_rate(self):
        """ Currently selected sample rate. """
        value = api.tiepie_hw_oscilloscope_get_sample_rate(self._handle)
        library.check_last_status_raise_on_error()
        return value

    def _set_sample_rate(self, value):
        api.tiepie_hw_oscilloscope_set_sample_rate(self._handle, value)
        library.check_last_status_raise_on_error()

    def verify_sample_rate(self, value):
        """ Verify if a required sample rate can be set, without actually setting the hardware itself.

        :param value: The required sample rate, in Hz.
        :returns: The sample rate that would have been set, if tiepie_hw_oscilloscope_set_sample_rate() was used.
        .. version added:: 1.0
        """
        result = api.tiepie_hw_oscilloscope_verify_sample_rate(self._handle, value)
        library.check_last_status_raise_on_error()
        return result

    def verify_sample_rate_ex(self, value, measure_mode, resolution, channel_enabled, channel_count):
        """ Verify sample rate by measure mode, resolution and active channels.

        :param value: The required sample rate, in Hz.
        :param measure_mode: Measure mode, a TIEPIE_HW_MM_* value.
        :param resolution: Resolution in bits.
        :param channel_enabled: Pointer to buffer with channel enables.
        :param channel_count: Number of items in ``channel_enabled.``
        :returns: Sample rate in Hz when set.
        .. version added:: 1.0
        """
        result = api.tiepie_hw_oscilloscope_verify_sample_rate_ex(self._handle, value, measure_mode, resolution, channel_enabled, channel_count)
        library.check_last_status_raise_on_error()
        return result

    def verify_sample_rates_ex(self, values, count, measure_mode, auto_resolution_mode, resolution, channel_enabled, channel_count):
        """ Verify sample rates by measure mode, resolution mode, resolution and active channels.

        :param values: Pointer to buffer with sample frequencies.
        :param count: Number of items in ``values.``
        :param measure_mode: Measure mode, a TIEPIE_HW_MM_* value.
        :param auto_resolution_mode: Auto resolution mode, a TIEPIE_HW_ARM_* value.
        :param resolution: Resolution in bits.
        :param channel_enabled: Pointer to buffer with channel enables.
        :param channel_count: Number of items in ``channel_enabled.``
        .. version added:: 1.0
        """
        api.tiepie_hw_oscilloscope_verify_sample_rates_ex(self._handle, values, count, measure_mode, auto_resolution_mode, resolution, channel_enabled, channel_count)
        library.check_last_status_raise_on_error()

    def _get_record_length_max(self):
        """  """
        value = api.tiepie_hw_oscilloscope_get_record_length_max(self._handle)
        library.check_last_status_raise_on_error()
        return value

    def get_record_length_max_ex(self, measure_mode, resolution):
        """ Get maximum record length for a specified measure mode and resolution.

        :param measure_mode: Measure mode, a TIEPIE_HW_MM_* value.
        :param resolution: Resolution in bits.
        :returns: _max_imum record length.
        .. version added:: 1.0
        """
        result = api.tiepie_hw_oscilloscope_get_record_length_max_ex(self._handle, measure_mode, resolution)
        library.check_last_status_raise_on_error()
        return result

    def _get_record_length(self):
        """ Currently selected record length. """
        value = api.tiepie_hw_oscilloscope_get_record_length(self._handle)
        library.check_last_status_raise_on_error()
        return value

    def _set_record_length(self, value):
        api.tiepie_hw_oscilloscope_set_record_length(self._handle, value)
        library.check_last_status_raise_on_error()

    def verify_record_length(self, record_length):
        """ Verify if a required record length can be set, without actually setting the hardware itself.

        :param record_length: The required record length, in samples.
        :returns: The record length that would have been set, if tiepie_hw_oscilloscope_set_record_length() was used.
        .. version added:: 1.0
        """
        result = api.tiepie_hw_oscilloscope_verify_record_length(self._handle, record_length)
        library.check_last_status_raise_on_error()
        return result

    def verify_record_length_ex(self, record_length, measure_mode, resolution, channel_enabled, channel_count):
        """ Verify record length by measure mode, resolution and active channels.

        :param record_length: Record length.
        :param measure_mode: Measure mode, a TIEPIE_HW_MM_* value.
        :param resolution: Resolution in bits.
        :param channel_enabled: Pointer to buffer with channel enables.
        :param channel_count: Number of items in ``channel_enabled.``
        :returns: Record length.
        .. version added:: 1.0
        """
        result = api.tiepie_hw_oscilloscope_verify_record_length_ex(self._handle, record_length, measure_mode, resolution, channel_enabled, channel_count)
        library.check_last_status_raise_on_error()
        return result

    def _get_pre_sample_ratio(self):
        """  """
        value = api.tiepie_hw_oscilloscope_get_pre_sample_ratio(self._handle)
        library.check_last_status_raise_on_error()
        return value

    def _set_pre_sample_ratio(self, value):
        api.tiepie_hw_oscilloscope_set_pre_sample_ratio(self._handle, value)
        library.check_last_status_raise_on_error()

    def _get_segment_count_max(self):
        """  """
        value = api.tiepie_hw_oscilloscope_get_segment_count_max(self._handle)
        library.check_last_status_raise_on_error()
        return value

    def get_segment_count_max_ex(self, measure_mode):
        """ Get the maximum supported number of segments for a specified measure mode.

        :param measure_mode: Measure mode, a TIEPIE_HW_MM_* value.
        :returns: The maximum supported number of segments, or ``0`` when unsuccessful.
        .. version added:: 1.0
        """
        result = api.tiepie_hw_oscilloscope_get_segment_count_max_ex(self._handle, measure_mode)
        library.check_last_status_raise_on_error()
        return result

    def _get_segment_count(self):
        """ Currently selected number of segments. """
        value = api.tiepie_hw_oscilloscope_get_segment_count(self._handle)
        library.check_last_status_raise_on_error()
        return value

    def _set_segment_count(self, value):
        api.tiepie_hw_oscilloscope_set_segment_count(self._handle, value)
        library.check_last_status_raise_on_error()

    def verify_segment_count(self, value):
        """ Verify if a required number of segments can be set, without actually setting the hardware itself.

        :param value: The required number of segments.
        :returns: The actually number of segments that would have been set, if tiepie_hw_oscilloscope_set_segment_count() was used.
        .. version added:: 1.0
        """
        result = api.tiepie_hw_oscilloscope_verify_segment_count(self._handle, value)
        library.check_last_status_raise_on_error()
        return result

    def verify_segment_count_ex(self, value, measure_mode, record_length, channel_enabled, channel_count):
        """ Verify number of segments by measure mode, record length and enabled channels.

        :param value: The required number of segments.
        :param measure_mode: Measure mode, a TIEPIE_HW_MM_* value.
        :param record_length: Record length in samples.
        :param channel_enabled: Pointer to buffer with channel enables.
        :param channel_count: Number of items in ``channel_enabled.``
        :returns: The actually number of segments that would have been set.
        .. version added:: 1.0
        """
        result = api.tiepie_hw_oscilloscope_verify_segment_count_ex(self._handle, value, measure_mode, record_length, channel_enabled, channel_count)
        library.check_last_status_raise_on_error()
        return result

    def _get_has_trigger(self):
        """  """
        value = api.tiepie_hw_oscilloscope_has_trigger(self._handle)
        library.check_last_status_raise_on_error()
        return value != BOOL_FALSE

    def has_trigger_ex(self, measure_mode):
        """ Check whether the oscilloscope has trigger support for a specified measure mode.

        :param measure_mode: Measure mode, a TIEPIE_HW_MM_* value.
        :returns: ``True`` if the oscilloscope has trigger support, ``False`` otherwise.
        .. version added:: 1.0
        """
        result = api.tiepie_hw_oscilloscope_has_trigger_ex(self._handle, measure_mode)
        library.check_last_status_raise_on_error()
        return result != BOOL_FALSE

    def _get_has_presamples_valid(self):
        """  """
        value = api.tiepie_hw_oscilloscope_has_presamples_valid(self._handle)
        library.check_last_status_raise_on_error()
        return value != BOOL_FALSE

    def has_presamples_valid_ex(self, measure_mode):
        """ Check whether the oscilloscope has presamples valid support for a specific measure mode.

        :param measure_mode: Measure mode, a TIEPIE_HW_MM_* value.
        :returns: ``True`` if the oscilloscope has presamples valid support, ``False`` otherwise.
        .. version added:: 1.0.1
        """
        result = api.tiepie_hw_oscilloscope_has_presamples_valid_ex(self._handle, measure_mode)
        library.check_last_status_raise_on_error()
        return result != BOOL_FALSE

    def _get_presamples_valid(self):
        """ Get presamples valid for a specified measure mode. """
        value = api.tiepie_hw_oscilloscope_get_presamples_valid(self._handle)
        library.check_last_status_raise_on_error()
        return value != BOOL_FALSE

    def _set_presamples_valid(self, value):
        value = BOOL_TRUE if value else BOOL_FALSE
        api.tiepie_hw_oscilloscope_set_presamples_valid(self._handle, value)
        library.check_last_status_raise_on_error()

    def _get_has_sureconnect(self):
        """  """
        value = api.tiepie_hw_oscilloscope_has_sureconnect(self._handle)
        library.check_last_status_raise_on_error()
        return value != BOOL_FALSE

    def start_sureconnect(self):
        """ Perform a SureConnect connection test on all enabled channels.

        .. version added:: 1.0
        """
        api.tiepie_hw_oscilloscope_start_sureconnect(self._handle)
        library.check_last_status_raise_on_error()

    def start_sureconnect_ex(self, channel_enabled, channel_count):
        """ Perform a SureConnect connection test on all channels.

        :param channel_enabled: A pointer to a buffer with channel enables.
        :param channel_count: The number of items in ``channel_enabled.``
        .. version added:: 1.0
        """
        api.tiepie_hw_oscilloscope_start_sureconnect_ex(self._handle, channel_enabled, channel_count)
        library.check_last_status_raise_on_error()

    def _get_is_sureconnect_completed(self):
        """ Check whether the SureConnect connection test on a specified oscilloscope is completed. """
        value = api.tiepie_hw_oscilloscope_is_sureconnect_completed(self._handle)
        library.check_last_status_raise_on_error()
        return value != BOOL_FALSE

    def get_sureconnect_data(self):
        """ Get the SureConnect connection test result data.

        :returns: :class:`list` of :class:`.TriState` values.
        .. version added:: 1.0
        """
        if not self.is_sureconnect_completed:
            raise UnsuccessfulError()

        channel_count = len(self.channels)

        # Allocate memory:
        buffer = (c_uint8 * channel_count)()

        # Get the data:
        channel_count = api.tiepie_hw_oscilloscope_get_sureconnect_data(self._handle, buffer, channel_count)
        library.check_last_status_raise_on_error()

        # Create result array:
        result = []
        for i in range(channel_count):
            result.append(convert_tristate(buffer[i]))

        return result

    is_demo = property(_get_is_demo)
    valid_pre_sample_count = property(_get_valid_pre_sample_count)
    measure_modes = property(_get_measure_modes)
    measure_mode = property(_get_measure_mode, _set_measure_mode)
    is_running = property(_get_is_running)
    is_triggered = property(_get_is_triggered)
    is_timeout_triggered = property(_get_is_timeout_triggered)
    is_force_triggered = property(_get_is_force_triggered)
    is_data_ready = property(_get_is_data_ready)
    is_data_overflow = property(_get_is_data_overflow)
    auto_resolution_modes = property(_get_auto_resolution_modes)
    auto_resolution_mode = property(_get_auto_resolution_mode, _set_auto_resolution_mode)
    resolutions = property(_get_resolutions)
    resolution = property(_get_resolution, _set_resolution)
    is_resolution_enhanced = property(_get_is_resolution_enhanced)
    clock_sources = property(_get_clock_sources)
    clock_source = property(_get_clock_source, _set_clock_source)
    clock_source_frequencies = property(_get_clock_source_frequencies)
    clock_source_frequency = property(_get_clock_source_frequency, _set_clock_source_frequency)
    clock_outputs = property(_get_clock_outputs)
    clock_output = property(_get_clock_output, _set_clock_output)
    clock_output_frequencies = property(_get_clock_output_frequencies)
    clock_output_frequency = property(_get_clock_output_frequency, _set_clock_output_frequency)
    sample_rate_max = property(_get_sample_rate_max)
    sample_rate = property(_get_sample_rate, _set_sample_rate)
    record_length_max = property(_get_record_length_max)
    record_length = property(_get_record_length, _set_record_length)
    pre_sample_ratio = property(_get_pre_sample_ratio, _set_pre_sample_ratio)
    segment_count_max = property(_get_segment_count_max)
    segment_count = property(_get_segment_count, _set_segment_count)
    has_trigger = property(_get_has_trigger)
    has_presamples_valid = property(_get_has_presamples_valid)
    presamples_valid = property(_get_presamples_valid, _set_presamples_valid)
    has_sureconnect = property(_get_has_sureconnect)
    is_sureconnect_completed = property(_get_is_sureconnect_completed)
    channels = property(_get_channels)
    trigger = property(_get_trigger)
