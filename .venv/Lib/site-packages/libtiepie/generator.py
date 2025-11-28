""" python-libtiepie - Python interface for libtiepie-hw library

Copyright (c) 2025 TiePie engineering

Website: https://www.tiepie.com/LibTiePie

"""

from array import array
from ctypes import c_void_p, c_double
from .api import api
from .const import *
from .library import library
from .device import Device


class Generator(Device):
    """"""

    def __init__(self, handle):
        super(Generator, self).__init__(handle)

    def _get_connector_type(self):
        """  """
        value = api.tiepie_hw_generator_get_connector_type(self._handle)
        library.check_last_status_raise_on_error()
        return value

    def _get_is_differential(self):
        """ Check whether the output is differential. """
        value = api.tiepie_hw_generator_is_differential(self._handle)
        library.check_last_status_raise_on_error()
        return value != BOOL_FALSE

    def _get_impedance(self):
        """ Output impedance. """
        value = api.tiepie_hw_generator_get_impedance(self._handle)
        library.check_last_status_raise_on_error()
        return value

    def _get_resolution(self):
        """ DAC resolution. """
        value = api.tiepie_hw_generator_get_resolution(self._handle)
        library.check_last_status_raise_on_error()
        return value

    def _get_output_value_min(self):
        """ Minimum output value. """
        value = api.tiepie_hw_generator_get_output_value_min(self._handle)
        library.check_last_status_raise_on_error()
        return value

    def _get_output_value_max(self):
        """ Maximum output value. """
        value = api.tiepie_hw_generator_get_output_value_max(self._handle)
        library.check_last_status_raise_on_error()
        return value

    def get_output_value_min_max(self, min, max):
        """ Get the minimum and/or maximum output value.

        :param min: A pointer to a memory location for the minimum value, or ``None.``
        :param max: A pointer to a memory location for the maximum value, or ``None.``
        .. version added:: 1.0
        """
        api.tiepie_hw_generator_get_output_value_min_max(self._handle, min, max)
        library.check_last_status_raise_on_error()

    def _get_is_controllable(self):
        """  """
        value = api.tiepie_hw_generator_is_controllable(self._handle)
        library.check_last_status_raise_on_error()
        return value != BOOL_FALSE

    def _get_is_running(self):
        """ Check whether the generator is running. """
        value = api.tiepie_hw_generator_is_running(self._handle)
        library.check_last_status_raise_on_error()
        return value != BOOL_FALSE

    def _get_status(self):
        """ Current signal generation status """
        value = api.tiepie_hw_generator_get_status(self._handle)
        library.check_last_status_raise_on_error()
        return value

    def _get_output_enable(self):
        """ Check whether a specified generator is enabled """
        value = api.tiepie_hw_generator_get_output_enable(self._handle)
        library.check_last_status_raise_on_error()
        return value != BOOL_FALSE

    def _set_output_enable(self, value):
        value = BOOL_TRUE if value else BOOL_FALSE
        api.tiepie_hw_generator_set_output_enable(self._handle, value)
        library.check_last_status_raise_on_error()

    def _get_has_output_invert(self):
        """ Check whether the output can be inverted """
        value = api.tiepie_hw_generator_has_output_invert(self._handle)
        library.check_last_status_raise_on_error()
        return value != BOOL_FALSE

    def _get_output_invert(self):
        """ Check whether the output is inverted """
        value = api.tiepie_hw_generator_get_output_invert(self._handle)
        library.check_last_status_raise_on_error()
        return value != BOOL_FALSE

    def _set_output_invert(self, value):
        value = BOOL_TRUE if value else BOOL_FALSE
        api.tiepie_hw_generator_set_output_invert(self._handle, value)
        library.check_last_status_raise_on_error()

    def start(self):
        """ Start the signal generation.

        .. version added:: 1.0
        """
        api.tiepie_hw_generator_start(self._handle)
        library.check_last_status_raise_on_error()

    def stop(self):
        """ Stop the signal generation.

        .. version added:: 1.0
        """
        api.tiepie_hw_generator_stop(self._handle)
        library.check_last_status_raise_on_error()

    def _get_signal_types(self):
        """  """
        value = api.tiepie_hw_generator_get_signal_types(self._handle)
        library.check_last_status_raise_on_error()
        return value

    def _get_signal_type(self):
        """ Currently selected signal type. """
        value = api.tiepie_hw_generator_get_signal_type(self._handle)
        library.check_last_status_raise_on_error()
        return value

    def _set_signal_type(self, value):
        api.tiepie_hw_generator_set_signal_type(self._handle, value)
        library.check_last_status_raise_on_error()

    def _get_has_amplitude(self):
        """  """
        value = api.tiepie_hw_generator_has_amplitude(self._handle)
        library.check_last_status_raise_on_error()
        return value != BOOL_FALSE

    def has_amplitude_ex(self, signal_type):
        """ Check whether the specified generator supports controlling the signal amplitude for a specified signal type.

        :param signal_type: The requested signal type, a TIEPIE_HW_ST_* value.
        :returns: ``True`` if supported, ``False`` if not.
        .. version added:: 1.0
        """
        result = api.tiepie_hw_generator_has_amplitude_ex(self._handle, signal_type)
        library.check_last_status_raise_on_error()
        return result != BOOL_FALSE

    def _get_amplitude_min(self):
        """ Minimum signal amplitude for the current signal type. """
        value = api.tiepie_hw_generator_get_amplitude_min(self._handle)
        library.check_last_status_raise_on_error()
        return value

    def _get_amplitude_max(self):
        """ Maximum signal amplitude for the current signal type. """
        value = api.tiepie_hw_generator_get_amplitude_max(self._handle)
        library.check_last_status_raise_on_error()
        return value

    def get_amplitude_min_max_ex(self, signal_type, min, max):
        """ Get the minimum and/or maximum amplitude for a specified signal type.

        :param signal_type: The requested signal type, a TIEPIE_HW_ST_* value.
        :param min: A pointer to a memory location for the minimum amplitude in Volt, or ``None.``
        :param max: A pointer to a memory location for the maximum amplitude in Volt, or ``None.``
        .. version added:: 1.0
        """
        api.tiepie_hw_generator_get_amplitude_min_max_ex(self._handle, signal_type, min, max)
        library.check_last_status_raise_on_error()

    def _get_amplitude(self):
        """ Currently set signal amplitude. """
        value = api.tiepie_hw_generator_get_amplitude(self._handle)
        library.check_last_status_raise_on_error()
        return value

    def _set_amplitude(self, value):
        api.tiepie_hw_generator_set_amplitude(self._handle, value)
        library.check_last_status_raise_on_error()

    def verify_amplitude(self, amplitude):
        """ Verify if a signal amplitude can be set, without actually setting the hardware itself.

        :param amplitude: The requested signal amplitude.
        :returns: The signal amplitude that would have been set, in Volt.
        .. version added:: 1.0
        """
        result = api.tiepie_hw_generator_verify_amplitude(self._handle, amplitude)
        library.check_last_status_raise_on_error()
        return result

    def verify_amplitude_ex(self, amplitude, signal_type, amplitude_range_index, offset, output_invert):
        """ Verify if a signal amplitude can be set for a specified signal type, amplitude range and offset, without actually setting the hardware itself.

        :param amplitude: The requested signal amplitude.
        :param signal_type: The requested signal type, a TIEPIE_HW_ST_* value.
        :param amplitude_range_index: The requested output range index or \ref TIEPIE_HW_RANGEINDEX_AUTO.
        :param offset: The requested signal offset.
        :param output_invert: The requested signal outputInvert.
        :returns: The signal amplitude that would have been set, in Volt.
        .. version added:: 1.0
        """
        output_invert = BOOL_TRUE if output_invert else BOOL_FALSE
        result = api.tiepie_hw_generator_verify_amplitude_ex(self._handle, amplitude, signal_type, amplitude_range_index, offset, output_invert)
        library.check_last_status_raise_on_error()
        return result

    def _get_amplitude_ranges(self):
        """  """
        count = api.tiepie_hw_generator_get_amplitude_ranges(self._handle, None, 0)
        library.check_last_status_raise_on_error()
        values = (c_double * count)()
        api.tiepie_hw_generator_get_amplitude_ranges(self._handle, values, count)
        library.check_last_status_raise_on_error()
        return array('d', values)

    def _get_amplitude_range(self):
        """ Currently set amplitude range. """
        value = api.tiepie_hw_generator_get_amplitude_range(self._handle)
        library.check_last_status_raise_on_error()
        return value

    def _set_amplitude_range(self, value):
        api.tiepie_hw_generator_set_amplitude_range(self._handle, value)
        library.check_last_status_raise_on_error()

    def _get_amplitude_auto_ranging(self):
        """ Amplitude auto ranging setting. """
        value = api.tiepie_hw_generator_get_amplitude_auto_ranging(self._handle)
        library.check_last_status_raise_on_error()
        return value != BOOL_FALSE

    def _set_amplitude_auto_ranging(self, value):
        value = BOOL_TRUE if value else BOOL_FALSE
        api.tiepie_hw_generator_set_amplitude_auto_ranging(self._handle, value)
        library.check_last_status_raise_on_error()

    def _get_has_offset(self):
        """  """
        value = api.tiepie_hw_generator_has_offset(self._handle)
        library.check_last_status_raise_on_error()
        return value != BOOL_FALSE

    def has_offset_ex(self, signal_type):
        """ Check whether the specified generator supports controlling the signal offset for a specified signal type.

        :param signal_type: The requested signal type, a TIEPIE_HW_ST_* value.
        :returns: ``True`` if supported, ``False`` if not.
        .. version added:: 1.0
        """
        result = api.tiepie_hw_generator_has_offset_ex(self._handle, signal_type)
        library.check_last_status_raise_on_error()
        return result != BOOL_FALSE

    def _get_offset_min(self):
        """ Minimum offset for the current signal type. """
        value = api.tiepie_hw_generator_get_offset_min(self._handle)
        library.check_last_status_raise_on_error()
        return value

    def _get_offset_max(self):
        """ Maximum offset for the current signal type. """
        value = api.tiepie_hw_generator_get_offset_max(self._handle)
        library.check_last_status_raise_on_error()
        return value

    def get_offset_min_max_ex(self, signal_type, min, max):
        """ Get the minimum and maximum offset for a specified signal type.

        :param signal_type: The requested signal type, a TIEPIE_HW_ST_* value.
        :param min: A pointer to a memory location for the minimum offset in Volt, or ``None.``
        :param max: A pointer to a memory location for the maximum offset in Volt, or ``None.``
        .. version added:: 1.0
        """
        api.tiepie_hw_generator_get_offset_min_max_ex(self._handle, signal_type, min, max)
        library.check_last_status_raise_on_error()

    def _get_offset(self):
        """ Current signal offset. """
        value = api.tiepie_hw_generator_get_offset(self._handle)
        library.check_last_status_raise_on_error()
        return value

    def _set_offset(self, value):
        api.tiepie_hw_generator_set_offset(self._handle, value)
        library.check_last_status_raise_on_error()

    def verify_offset(self, value):
        """ Verify if a signal offset can be set, without actually setting the hardware itself.

        :param value: The requested signal offset, in Volt.
        :returns: The signal offset that would have been set, in Volt.
        .. version added:: 1.0
        """
        result = api.tiepie_hw_generator_verify_offset(self._handle, value)
        library.check_last_status_raise_on_error()
        return result

    def verify_offset_ex(self, value, signal_type, amplitude, output_invert):
        """ Verify if a signal offset can be set for a specified signal type and amplitude, without actually setting the hardware itself.

        :param value: The requested signal offset.
        :param signal_type: The requested signal type, a TIEPIE_HW_ST_* value.
        :param amplitude: The requested signal amplitude, ignored for #TIEPIE_HW_ST_dC.
        :param output_invert: The requested output_invert.
        :returns: The signal offset that would have been set, in Volt.
        .. version added:: 1.0
        """
        output_invert = BOOL_TRUE if output_invert else BOOL_FALSE
        result = api.tiepie_hw_generator_verify_offset_ex(self._handle, value, signal_type, amplitude, output_invert)
        library.check_last_status_raise_on_error()
        return result

    def _get_frequency_modes(self):
        """  """
        value = api.tiepie_hw_generator_get_frequency_modes(self._handle)
        library.check_last_status_raise_on_error()
        return value

    def get_frequency_modes_ex(self, signal_type):
        """ Get the supported generator frequency modes for a specified signal type.

        :param signal_type: The requested signal type, a TIEPIE_HW_ST_* value.
        :returns: The supported generator frequency modes for th especified signal type, a set of OR-ed TIEPIE_HW_FM_* values, or #TIEPIE_HW_FMM_NONE.
        .. version added:: 1.0
        """
        result = api.tiepie_hw_generator_get_frequency_modes_ex(self._handle, signal_type)
        library.check_last_status_raise_on_error()
        return result

    def _get_frequency_mode(self):
        """ Current generator frequency mode """
        value = api.tiepie_hw_generator_get_frequency_mode(self._handle)
        library.check_last_status_raise_on_error()
        return value

    def _set_frequency_mode(self, value):
        api.tiepie_hw_generator_set_frequency_mode(self._handle, value)
        library.check_last_status_raise_on_error()

    def _get_has_frequency(self):
        """ Check whether the current signal type and frequency mode support controlling the signal/sample frequency. """
        value = api.tiepie_hw_generator_has_frequency(self._handle)
        library.check_last_status_raise_on_error()
        return value != BOOL_FALSE

    def has_frequency_ex(self, frequency_mode, signal_type):
        """ Check whether the specified generator supports controlling the signal/sample frequency for the specified frequency mode and signal type.

        :param frequency_mode: The requested generator frequency mode, a TIEPIE_HW_FM_* value.
        :param signal_type: The requested signal type, a TIEPIE_HW_ST_* value.
        :returns: ``True`` if supported, ``False`` if not.
        .. version added:: 1.0
        """
        result = api.tiepie_hw_generator_has_frequency_ex(self._handle, frequency_mode, signal_type)
        library.check_last_status_raise_on_error()
        return result != BOOL_FALSE

    def _get_frequency_min(self):
        """ Minimum signal/sample frequency with the current frequency mode. """
        value = api.tiepie_hw_generator_get_frequency_min(self._handle)
        library.check_last_status_raise_on_error()
        return value

    def _get_frequency_max(self):
        """ Maximum signal/sample frequency with the current frequency mode and signal type. """
        value = api.tiepie_hw_generator_get_frequency_max(self._handle)
        library.check_last_status_raise_on_error()
        return value

    def get_frequency_min_max(self, frequency_mode, min, max):
        """ Get the minimum and maximum signal/sample frequency for a specified frequency mode and the current signal type.

        :param frequency_mode: The requested generator frequency mode, a TIEPIE_HW_FM_* value.
        :param min: A pointer to a memory location for the minimum frequency, or ``None.``
        :param max: A pointer to a memory location for the maximum frequency, or ``None.``
        .. version added:: 1.0
        """
        api.tiepie_hw_generator_get_frequency_min_max(self._handle, frequency_mode, min, max)
        library.check_last_status_raise_on_error()

    def get_frequency_min_max_ex(self, frequency_mode, signal_type, min, max):
        """ Get the minimum and maximum signal/sample frequency for a specified frequency mode and signal type.

        :param frequency_mode: The requested generator frequency mode, a TIEPIE_HW_FM_* value.
        :param signal_type: The requested signal type, a TIEPIE_HW_ST_* value.
        :param min: A pointer to a memory location for the minimum frequency, or ``None.``
        :param max: A pointer to a memory location for the maximum frequency, or ``None.``
        .. version added:: 1.0
        """
        api.tiepie_hw_generator_get_frequency_min_max_ex(self._handle, frequency_mode, signal_type, min, max)
        library.check_last_status_raise_on_error()

    def _get_frequency(self):
        """ Current signal/sample frequency. """
        value = api.tiepie_hw_generator_get_frequency(self._handle)
        library.check_last_status_raise_on_error()
        return value

    def _set_frequency(self, value):
        api.tiepie_hw_generator_set_frequency(self._handle, value)
        library.check_last_status_raise_on_error()

    def verify_frequency(self, value):
        """ Verify if a signal/sample frequency can be set, without actually setting the hardware itself.

        :param value: The requested signal/sample rate, in Hz.
        :returns: The signal/sample rate that would have been set, in Hz.
        .. version added:: 1.0
        """
        result = api.tiepie_hw_generator_verify_frequency(self._handle, value)
        library.check_last_status_raise_on_error()
        return result

    def verify_frequency_ex(self, value, frequency_mode, signal_type, data_length, width):
        """ Verify if a signal/sample frequency can be set for a specified frequency mode, signal type and arbitrary waveform pattern length, without actually setting the hardware itself.

        :param value: The requested signal/sample frequency, in Hz.
        :param frequency_mode: The requested generator frequency mode, a TIEPIE_HW_FM_* value.
        :param signal_type: The requested signal type, a TIEPIE_HW_ST_* value.
        :param data_length: The requested Arbitrary waveform pattern length.
        :param width: Pulse width in seconds, only for #TIEPIE_HW_ST_PULSE.
        :returns: The signal/sample frequency that would have been set, in Hz.
        .. version added:: 1.0
        """
        result = api.tiepie_hw_generator_verify_frequency_ex(self._handle, value, frequency_mode, signal_type, data_length, width)
        library.check_last_status_raise_on_error()
        return result

    def _get_has_phase(self):
        """  """
        value = api.tiepie_hw_generator_has_phase(self._handle)
        library.check_last_status_raise_on_error()
        return value != BOOL_FALSE

    def has_phase_ex(self, signal_type):
        """ Check whether the specified generator supports controlling the signal phase for a specified signal type.

        :param signal_type: The requested signal type, a TIEPIE_HW_ST_* value.
        :returns: ``True`` if supported, ``False`` if not.
        .. version added:: 1.0
        """
        result = api.tiepie_hw_generator_has_phase_ex(self._handle, signal_type)
        library.check_last_status_raise_on_error()
        return result != BOOL_FALSE

    def _get_phase_min(self):
        """ Minimum signal phase. """
        value = api.tiepie_hw_generator_get_phase_min(self._handle)
        library.check_last_status_raise_on_error()
        return value

    def _get_phase_max(self):
        """ Maximum signal phase. """
        value = api.tiepie_hw_generator_get_phase_max(self._handle)
        library.check_last_status_raise_on_error()
        return value

    def get_phase_min_max_ex(self, signal_type, min, max):
        """ Get the minimum and maximum phase for a specified signal type.

        :param signal_type: The requested signal type, a TIEPIE_HW_ST_* value.
        :param min: A pointer to a memory location for the minimum phase, or ``None.``
        :param max: A pointer to a memory location for the maximum phase, or ``None.``
        .. version added:: 1.0
        """
        api.tiepie_hw_generator_get_phase_min_max_ex(self._handle, signal_type, min, max)
        library.check_last_status_raise_on_error()

    def _get_phase(self):
        """ Current signal phase. """
        value = api.tiepie_hw_generator_get_phase(self._handle)
        library.check_last_status_raise_on_error()
        return value

    def _set_phase(self, value):
        api.tiepie_hw_generator_set_phase(self._handle, value)
        library.check_last_status_raise_on_error()

    def verify_phase(self, value):
        """ Verify if a phase can be set, without actually setting the hardware itself.

        :param value: The requested signal phase, a number between ``0`` and ``1.``
        :returns: The signal phase that would have been set, a number between ``0`` and ``1.``
        .. version added:: 1.0
        """
        result = api.tiepie_hw_generator_verify_phase(self._handle, value)
        library.check_last_status_raise_on_error()
        return result

    def verify_phase_ex(self, value, signal_type):
        """ Verify if a phase can be set for a specific signal type, without actually setting the hardware itself.

        :param value: The requested signal phase, a number between ``0`` and ``1.``
        :param signal_type: The requested signal type, a TIEPIE_HW_ST_* value.
        :returns: The signal phase that would have been set, a number between ``0`` and ``1.``
        .. version added:: 1.0
        """
        result = api.tiepie_hw_generator_verify_phase_ex(self._handle, value, signal_type)
        library.check_last_status_raise_on_error()
        return result

    def _get_has_symmetry(self):
        """  """
        value = api.tiepie_hw_generator_has_symmetry(self._handle)
        library.check_last_status_raise_on_error()
        return value != BOOL_FALSE

    def has_symmetry_ex(self, signal_type):
        """ Check whether the specified generator supports controlling the signal symmetry for a specified signal type.

        :param signal_type: The requested signal type, a TIEPIE_HW_ST_* value.
        :returns: ``True`` if supported, ``False`` if not.
        .. version added:: 1.0
        """
        result = api.tiepie_hw_generator_has_symmetry_ex(self._handle, signal_type)
        library.check_last_status_raise_on_error()
        return result != BOOL_FALSE

    def _get_symmetry_min(self):
        """ Minimum signal symmetry. """
        value = api.tiepie_hw_generator_get_symmetry_min(self._handle)
        library.check_last_status_raise_on_error()
        return value

    def _get_symmetry_max(self):
        """ Maximum signal symmetry. """
        value = api.tiepie_hw_generator_get_symmetry_max(self._handle)
        library.check_last_status_raise_on_error()
        return value

    def get_symmetry_min_max_ex(self, signal_type, min, max):
        """ Get the minimum and maximum symmetry for a specified signal type.

        :param signal_type: The requested signal type, a TIEPIE_HW_ST_* value.
        :param min: A pointer to a memory location for the minimum symmetry, or ``None.``
        :param max: A pointer to a memory location for the maximum symmetry, or ``None.``
        .. version added:: 1.0
        """
        api.tiepie_hw_generator_get_symmetry_min_max_ex(self._handle, signal_type, min, max)
        library.check_last_status_raise_on_error()

    def _get_symmetry(self):
        """ Current signal symmetry. """
        value = api.tiepie_hw_generator_get_symmetry(self._handle)
        library.check_last_status_raise_on_error()
        return value

    def _set_symmetry(self, value):
        api.tiepie_hw_generator_set_symmetry(self._handle, value)
        library.check_last_status_raise_on_error()

    def verify_symmetry(self, value):
        """ Verify if a symmetry can be set, without actually setting the hardware itself.

        :param value: The requested signal symmetry, a number between ``0`` and ``1.``
        :returns: The signal symmetry that would have been set, a number between ``0`` and ``1.``
        .. version added:: 1.0
        """
        result = api.tiepie_hw_generator_verify_symmetry(self._handle, value)
        library.check_last_status_raise_on_error()
        return result

    def verify_symmetry_ex(self, value, signal_type):
        """ Verify if a symmetry can be set for a specific signal type, without actually setting the hardware itself.

        :param value: The requested signal symmetry, a number between ``0`` and ``1.``
        :param signal_type: The requested signal type, a TIEPIE_HW_ST_* value.
        :returns: The signal symmetry that would have been set, a number between ``0`` and ``1.``
        .. version added:: 1.0
        """
        result = api.tiepie_hw_generator_verify_symmetry_ex(self._handle, value, signal_type)
        library.check_last_status_raise_on_error()
        return result

    def _get_has_width(self):
        """  """
        value = api.tiepie_hw_generator_has_width(self._handle)
        library.check_last_status_raise_on_error()
        return value != BOOL_FALSE

    def has_width_ex(self, signal_type):
        """ Check whether the specified generator supports controlling the signal pulse width for a specified signal type.

        :param signal_type: The requested signal type, a TIEPIE_HW_ST_* value.
        :returns: ``True`` if supported, ``False`` if not.
        .. version added:: 1.0
        """
        result = api.tiepie_hw_generator_has_width_ex(self._handle, signal_type)
        library.check_last_status_raise_on_error()
        return result != BOOL_FALSE

    def _get_width_min(self):
        """ Minimum pulse width with the current signal frequency. """
        value = api.tiepie_hw_generator_get_width_min(self._handle)
        library.check_last_status_raise_on_error()
        return value

    def _get_width_max(self):
        """ Maximum pulse width with the current signal frequency. """
        value = api.tiepie_hw_generator_get_width_max(self._handle)
        library.check_last_status_raise_on_error()
        return value

    def get_width_min_max_ex(self, signal_type, signal_frequency, min, max):
        """ Get the minimum and maximum pulse width for a specified signal type and signal frequency.

        :param signal_type: The requested signal type, a TIEPIE_HW_ST_* value.
        :param signal_frequency: The requested signal frequency in Hz.
        :param min: A pointer to a memory location for the minimum pulse width, or ``None.``
        :param max: A pointer to a memory location for the maximum pulse width, or ``None.``
        .. version added:: 1.0
        """
        api.tiepie_hw_generator_get_width_min_max_ex(self._handle, signal_type, signal_frequency, min, max)
        library.check_last_status_raise_on_error()

    def _get_width(self):
        """ Current pulse width. """
        value = api.tiepie_hw_generator_get_width(self._handle)
        library.check_last_status_raise_on_error()
        return value

    def _set_width(self, value):
        api.tiepie_hw_generator_set_width(self._handle, value)
        library.check_last_status_raise_on_error()

    def verify_width(self, value):
        """ Verify if a pulse width can be set, without actually setting the hardware itself.

        :param value: The requested pulse width in seconds.
        :returns: The pulse width that would have been set, in seconds.
        .. version added:: 1.0
        """
        result = api.tiepie_hw_generator_verify_width(self._handle, value)
        library.check_last_status_raise_on_error()
        return result

    def verify_width_ex(self, value, signal_type, signal_frequency):
        """ Verify if a pulse width can be set for a specific signal type and signal frequency, without actually setting the hardware itself.

        :param value: The requested pulse width in seconds.
        :param signal_type: The requested signal type, a TIEPIE_HW_ST_* value.
        :param signal_frequency: The requested signal frequency in Hz.
        :returns: pulse width in seconds.
        .. version added:: 1.0
        """
        result = api.tiepie_hw_generator_verify_width_ex(self._handle, value, signal_type, signal_frequency)
        library.check_last_status_raise_on_error()
        return result

    def _get_has_edge_time(self):
        """  """
        value = api.tiepie_hw_generator_has_edge_time(self._handle)
        library.check_last_status_raise_on_error()
        return value != BOOL_FALSE

    def has_edge_time_ex(self, signal_type):
        """ Check whether the specified generator supports controlling the edge times for a specified signal type.

        :param signal_type: The requested signal type, a TIEPIE_HW_ST_* value.
        :returns: ``True`` if supported, ``False`` if not.
        .. version added:: 1.0
        """
        result = api.tiepie_hw_generator_has_edge_time_ex(self._handle, signal_type)
        library.check_last_status_raise_on_error()
        return result != BOOL_FALSE

    def _get_leading_edge_time_min(self):
        """ Minimum leading edge time with the current pulse width and signal frequency. """
        value = api.tiepie_hw_generator_get_leading_edge_time_min(self._handle)
        library.check_last_status_raise_on_error()
        return value

    def _get_leading_edge_time_max(self):
        """ Maximum leading edge time with the current pulse width and signal frequency. """
        value = api.tiepie_hw_generator_get_leading_edge_time_max(self._handle)
        library.check_last_status_raise_on_error()
        return value

    def get_leading_edge_time_min_max_ex(self, signal_type, signal_frequency, symmetry, width, trailing_edge_time, min, max):
        """ Get the minimum and maximum leading edge time with the requested signal type, frequency, symmetry, pulse width and trailing edge time.

        :param signal_type: The requested signal type, a TIEPIE_HW_ST_* value.
        :param signal_frequency: The requested signal frequency in Hz.
        :param symmetry: The requested signal symmetry, a number between ``0`` and ``1.``
        :param width: The requested pulse width in seconds.
        :param trailing_edge_time: The requested trailing edge time in seconds.
        :param min: A pointer to a memory location for the minimum leading edge time, or ``None.``
        :param max: A pointer to a memory location for the maximum leading edge time, or ``None.``
        .. version added:: 1.0
        """
        api.tiepie_hw_generator_get_leading_edge_time_min_max_ex(self._handle, signal_type, signal_frequency, symmetry, width, trailing_edge_time, min, max)
        library.check_last_status_raise_on_error()

    def _get_leading_edge_time(self):
        """ Current leading edge time with the current pulse width and signal frequency. """
        value = api.tiepie_hw_generator_get_leading_edge_time(self._handle)
        library.check_last_status_raise_on_error()
        return value

    def _set_leading_edge_time(self, value):
        api.tiepie_hw_generator_set_leading_edge_time(self._handle, value)
        library.check_last_status_raise_on_error()

    def verify_leading_edge_time(self, leading_edge_time):
        """ Verify if a leading edge time can be set for the current signal type and signal frequency, without actually setting the hardware itself.

        :param leading_edge_time: The requested leading edge time in seconds.
        :returns: The leading edge time that would have been set, in seconds.
        .. version added:: 1.0
        """
        result = api.tiepie_hw_generator_verify_leading_edge_time(self._handle, leading_edge_time)
        library.check_last_status_raise_on_error()
        return result

    def verify_leading_edge_time_ex(self, leading_edge_time, signal_type, signal_frequency, symmetry, width, trailing_edge_time):
        """ Verify if a leading edge time can be set, with the specified signal type, frequency, symmetrym pulse width and trailing edge time, without actually setting the hardware itself.

        :param leading_edge_time: The requested leading edge time in seconds.
        :param signal_type: The requested signal type, a TIEPIE_HW_ST_* value.
        :param signal_frequency: The requested signal frequency in Hz.
        :param symmetry: The requested signal symmetry, a number between ``0`` and ``1.``
        :param width: The requested pulse width in seconds.
        :param trailing_edge_time: The requested trailing edge time in seconds.
        :returns: The leading edge time that would have been set, in seconds.
        .. version added:: 1.0
        """
        result = api.tiepie_hw_generator_verify_leading_edge_time_ex(self._handle, leading_edge_time, signal_type, signal_frequency, symmetry, width, trailing_edge_time)
        library.check_last_status_raise_on_error()
        return result

    def _get_trailing_edge_time_min(self):
        """ Minimum trailing edge time with the current pulse width and signal frequency. """
        value = api.tiepie_hw_generator_get_trailing_edge_time_min(self._handle)
        library.check_last_status_raise_on_error()
        return value

    def _get_trailing_edge_time_max(self):
        """ Maximum trailing edge time with the current pulse width and signal frequency. """
        value = api.tiepie_hw_generator_get_trailing_edge_time_max(self._handle)
        library.check_last_status_raise_on_error()
        return value

    def get_trailing_edge_time_min_max_ex(self, signal_type, signal_frequency, symmetry, width, leading_edge_time, min, max):
        """ Get the minimum and maximum trailing edge time with the requested signal type, frequency, symmetry, pulse width and leading edge time.

        :param signal_type: The requested signal type, a TIEPIE_HW_ST_* value.
        :param signal_frequency: The requested signal frequency in Hz.
        :param symmetry: The requested signal symmetry, a number between ``0`` and ``1.``
        :param width: The requested pulse width in seconds.
        :param leading_edge_time: The requested trailing edge time in seconds.
        :param min: A pointer to a memory location for the minimum trailing edge time, or ``None.``
        :param max: A pointer to a memory location for the maximum trailing edge time, or ``None.``
        .. version added:: 1.0
        """
        api.tiepie_hw_generator_get_trailing_edge_time_min_max_ex(self._handle, signal_type, signal_frequency, symmetry, width, leading_edge_time, min, max)
        library.check_last_status_raise_on_error()

    def _get_trailing_edge_time(self):
        """ Current trailing edge time with the current pulse width and signal frequency. """
        value = api.tiepie_hw_generator_get_trailing_edge_time(self._handle)
        library.check_last_status_raise_on_error()
        return value

    def _set_trailing_edge_time(self, value):
        api.tiepie_hw_generator_set_trailing_edge_time(self._handle, value)
        library.check_last_status_raise_on_error()

    def verify_trailing_edge_time(self, value):
        """ Verify if a trailing edge time can be set for the current signal type and signal frequency, without actually setting the hardware itself.

        :param value: The requested trailing edge time in seconds.
        :returns: The trailing edge time that would have been set, in seconds.
        .. version added:: 1.0
        """
        result = api.tiepie_hw_generator_verify_trailing_edge_time(self._handle, value)
        library.check_last_status_raise_on_error()
        return result

    def verify_trailing_edge_time_ex(self, trailing_edge_time, signal_type, signal_frequency, symmetry, width, leading_edge_time):
        """ Verify if a trailing edge time can be set, with the specified signal type, frequency, symmetrym pulse width and trailing edge time, without actually setting the hardware itself.

        :param trailing_edge_time: The requested trailing edge time in seconds.
        :param signal_type: The requested signal type, a TIEPIE_HW_ST_* value.
        :param signal_frequency: The requested signal frequency in Hz.
        :param symmetry: The requested signal symmetry, a number between ``0`` and ``1.``
        :param width: The requested pulse width in seconds.
        :param leading_edge_time: The requested leading edge time in seconds.
        :returns: The trailing edge time that would have been set, in seconds.
        .. version added:: 1.0
        """
        result = api.tiepie_hw_generator_verify_trailing_edge_time_ex(self._handle, trailing_edge_time, signal_type, signal_frequency, symmetry, width, leading_edge_time)
        library.check_last_status_raise_on_error()
        return result

    def _get_has_data(self):
        """  """
        value = api.tiepie_hw_generator_has_data(self._handle)
        library.check_last_status_raise_on_error()
        return value != BOOL_FALSE

    def has_data_ex(self, signal_type):
        """ Check whether the specified generator supports controlling the Arbitrary waveform buffer for a specified signal type.

        :param signal_type: The requested signal type, a TIEPIE_HW_ST_* value.
        :returns: ``True`` if supported, ``False`` if not.
        .. version added:: 1.0
        """
        result = api.tiepie_hw_generator_has_data_ex(self._handle, signal_type)
        library.check_last_status_raise_on_error()
        return result != BOOL_FALSE

    def _get_data_length_min(self):
        """ Minimum length of the waveform buffer. """
        value = api.tiepie_hw_generator_get_data_length_min(self._handle)
        library.check_last_status_raise_on_error()
        return value

    def _get_data_length_max(self):
        """ Maximum length of the waveform buffer. """
        value = api.tiepie_hw_generator_get_data_length_max(self._handle)
        library.check_last_status_raise_on_error()
        return value

    def get_data_length_min_max_ex(self, signal_type, min, max):
        """ Get the minimum and maximum length of the waveform buffer for a specified signal type.

        :param signal_type: The requested signal type, a TIEPIE_HW_ST_* value.
        :param min: A pointer to a memory location for the minimum data length, or ``None.``
        :param max: A pointer to a memory location for the maximum data length, or ``None.``
        .. version added:: 1.0
        """
        api.tiepie_hw_generator_get_data_length_min_max_ex(self._handle, signal_type, min, max)
        library.check_last_status_raise_on_error()

    def _get_data_length(self):
        """ Length of the currently loaded waveform pattern. """
        value = api.tiepie_hw_generator_get_data_length(self._handle)
        library.check_last_status_raise_on_error()
        return value

    def verify_data_length(self, value):
        """ Verify if a specified length of the waveform buffer for the current signal type can be set, without actually setting the hardware itself.

        :param value: The requested waveform buffer length in samples.
        :returns: The waveform buffer length that would have been set, in samples.
        .. version added:: 1.0
        """
        result = api.tiepie_hw_generator_verify_data_length(self._handle, value)
        library.check_last_status_raise_on_error()
        return result

    def verify_data_length_ex(self, value, signal_type):
        """ Verify if a specified length of the waveform buffer for a specified signal type can be set, without actually setting the hardware itself.

        :param value: The requested waveform buffer length in samples.
        :param signal_type: The requested signal type, a TIEPIE_HW_ST_* value.
        :returns: Waveform buffer length in samples.
        .. version added:: 1.0
        """
        result = api.tiepie_hw_generator_verify_data_length_ex(self._handle, value, signal_type)
        library.check_last_status_raise_on_error()
        return result

    def set_data(self, data):
        """ Load a waveform pattern into the waveform buffer.

        :param data: :class:`array.array` of floats, the waveform data.
        .. version added:: 1.0
        """
        if isinstance(data, array) and data.typecode == 'f':
            data_ptr = c_void_p(data.buffer_info()[0])
            data_ptr.__ref = data
            api.tiepie_hw_generator_set_data(self._handle, data_ptr, len(data))
            library.check_last_status_raise_on_error()
        else:
            raise Exception('Invalid data, must be array.array with typecode = `f`')

    def set_data_ex(self, buffer, sample_count, signal_type):
        """ Load a waveform pattern for a specified signal type into the waveform buffer.

        :param buffer: A pointer to a buffer with waveform data.
        :param sample_count: The number of samples in the buffer.
        :param signal_type: The requested signal type, a TIEPIE_HW_ST_* value.
        .. version added:: 1.0
        """
        api.tiepie_hw_generator_set_data_ex(self._handle, buffer, sample_count, signal_type)
        library.check_last_status_raise_on_error()

    def _get_data_raw_type(self):
        """  """
        value = api.tiepie_hw_generator_get_data_raw_type(self._handle)
        library.check_last_status_raise_on_error()
        return value

    def get_data_raw_value_range(self, min, zero, max):
        """ Get raw data minimum, equal to zero and maximum values.

        :param min: Pointer to buffer for possible minimum raw data value, or ``None.``
        :param zero: Pointer to buffer for equal to zero raw data value, or ``None.``
        :param max: Pointer to buffer for possible maximum raw data value, or ``None.``
        .. version added:: 1.0
        """
        api.tiepie_hw_generator_get_data_raw_value_range(self._handle, min, zero, max)
        library.check_last_status_raise_on_error()

    def _get_data_raw_value_min(self):
        """ Get maximum raw data value. """
        value = api.tiepie_hw_generator_get_data_raw_value_min(self._handle)
        library.check_last_status_raise_on_error()
        return value

    def _get_data_raw_value_zero(self):
        """ Get raw data value that equals zero. """
        value = api.tiepie_hw_generator_get_data_raw_value_zero(self._handle)
        library.check_last_status_raise_on_error()
        return value

    def _get_data_raw_value_max(self):
        """ Get minimum raw data value. """
        value = api.tiepie_hw_generator_get_data_raw_value_max(self._handle)
        library.check_last_status_raise_on_error()
        return value

    def set_data_raw(self, buffer, sample_count):
        """ Load a waveform pattern into the waveform buffer.

        :param buffer: Pointer to buffer with waveform data.
        :param sample_count: Number of samples in buffer.
        .. version added:: 1.0
        """
        api.tiepie_hw_generator_set_data_raw(self._handle, buffer, sample_count)
        library.check_last_status_raise_on_error()

    def set_data_raw_ex(self, buffer, sample_count, signal_type):
        """ Load a waveform pattern into the waveform buffer.

        :param buffer: Pointer to buffer with waveform data.
        :param sample_count: Number of samples in buffer.
        :param signal_type: Signal type, a TIEPIE_HW_ST_* value.
        .. version added:: 1.0
        """
        api.tiepie_hw_generator_set_data_raw_ex(self._handle, buffer, sample_count, signal_type)
        library.check_last_status_raise_on_error()

    def _get_modes(self):
        """  """
        value = api.tiepie_hw_generator_get_modes(self._handle)
        library.check_last_status_raise_on_error()
        return value

    def get_modes_ex(self, signal_type, frequency_mode):
        """ Get the supported generator modes for a specified signal type and frequency mode.

        :param signal_type: The requested signal type, a TIEPIE_HW_ST_* value.
        :param frequency_mode: The requested generator frequency mode, a TIEPIE_HW_FM_* value. (Ignored for #TIEPIE_HW_ST_dC)
        :returns: The supported generator modes, a set of OR-ed TIEPIE_HW_GM_* values or #TIEPIE_HW_GMM_NONE when unsuccessful.
        .. version added:: 1.0
        """
        result = api.tiepie_hw_generator_get_modes_ex(self._handle, signal_type, frequency_mode)
        library.check_last_status_raise_on_error()
        return result

    def _get_modes_native(self):
        """ :class:`array.array` of supported generator modes, regardless of the signal type and frequency mode. """
        value = api.tiepie_hw_generator_get_modes_native(self._handle)
        library.check_last_status_raise_on_error()
        return value

    def _get_mode(self):
        """ Current generator mode. """
        value = api.tiepie_hw_generator_get_mode(self._handle)
        library.check_last_status_raise_on_error()
        return value

    def _set_mode(self, value):
        api.tiepie_hw_generator_set_mode(self._handle, value)
        library.check_last_status_raise_on_error()

    def _get_is_burst_active(self):
        """  """
        value = api.tiepie_hw_generator_is_burst_active(self._handle)
        library.check_last_status_raise_on_error()
        return value != BOOL_FALSE

    def _get_burst_count_min(self):
        """ Minimum burst count for the current generator mode. """
        value = api.tiepie_hw_generator_get_burst_count_min(self._handle)
        library.check_last_status_raise_on_error()
        return value

    def _get_burst_count_max(self):
        """ Maximum burst count for the current generator mode. """
        value = api.tiepie_hw_generator_get_burst_count_max(self._handle)
        library.check_last_status_raise_on_error()
        return value

    def get_burst_count_min_max_ex(self, generator_mode, min, max):
        """ Get the minimum and maximum burst count mode.

        :param generator_mode: The requested generator mode, a TIEPIE_HW_GM_* value.
        :param min: A pointer to a memory location for the minimum or ``None.``
        :param max: A pointer to a memory location for the maximum or ``None.``
        .. version added:: 1.0
        """
        api.tiepie_hw_generator_get_burst_count_min_max_ex(self._handle, generator_mode, min, max)
        library.check_last_status_raise_on_error()

    def _get_burst_count(self):
        """ Current burst count for the current generator mode. """
        value = api.tiepie_hw_generator_get_burst_count(self._handle)
        library.check_last_status_raise_on_error()
        return value

    def _set_burst_count(self, value):
        api.tiepie_hw_generator_set_burst_count(self._handle, value)
        library.check_last_status_raise_on_error()

    def _get_burst_sample_count_min(self):
        """ Minimum burst sample count for the current generator mode. """
        value = api.tiepie_hw_generator_get_burst_sample_count_min(self._handle)
        library.check_last_status_raise_on_error()
        return value

    def _get_burst_sample_count_max(self):
        """ Maximum burst sample count for the current generator mode. """
        value = api.tiepie_hw_generator_get_burst_sample_count_max(self._handle)
        library.check_last_status_raise_on_error()
        return value

    def get_burst_sample_count_min_max_ex(self, generator_mode, min, max):
        """ Get the minimum and maximum burst sample count mode.

        :param generator_mode: The requested generator mode, a TIEPIE_HW_GM_* value.
        :param min: A pointer to a memory location for the minimum or ``None.``
        :param max: A pointer to a memory location for the maximum or ``None.``
        .. version added:: 1.0
        """
        api.tiepie_hw_generator_get_burst_sample_count_min_max_ex(self._handle, generator_mode, min, max)
        library.check_last_status_raise_on_error()

    def _get_burst_sample_count(self):
        """ Current burst sample count for the current generator mode. """
        value = api.tiepie_hw_generator_get_burst_sample_count(self._handle)
        library.check_last_status_raise_on_error()
        return value

    def _set_burst_sample_count(self, value):
        api.tiepie_hw_generator_set_burst_sample_count(self._handle, value)
        library.check_last_status_raise_on_error()

    def _get_burst_segment_count_min(self):
        """ Minimum burst segment count for the current settings. """
        value = api.tiepie_hw_generator_get_burst_segment_count_min(self._handle)
        library.check_last_status_raise_on_error()
        return value

    def _get_burst_segment_count_max(self):
        """ Maximum burst segment count for the current settings. """
        value = api.tiepie_hw_generator_get_burst_segment_count_max(self._handle)
        library.check_last_status_raise_on_error()
        return value

    def get_burst_segment_count_min_max_ex(self, generator_mode, signal_type, frequency_mode, frequency, data_length, min, max):
        """ Get the minimum and maximum burst segment count mode, signal type, frequency mode, frequency and data length.

        :param generator_mode: The requested generator mode, a TIEPIE_HW_GM_* value.
        :param signal_type: The requested signal type, a TIEPIE_HW_ST_* value.
        :param frequency_mode: The requested frequency mode, a TIEPIE_HW_FM_* value. (Ignored for #TIEPIE_HW_ST_dC)
        :param frequency: The requested frequency in Hz.
        :param data_length: The requested data length in samples, only for #TIEPIE_HW_ST_ARBITRARY.
        :param min: A pointer to a memory location for the minimum or ``None.``
        :param max: A pointer to a memory location for the maximum or ``None.``
        .. version added:: 1.0
        """
        api.tiepie_hw_generator_get_burst_segment_count_min_max_ex(self._handle, generator_mode, signal_type, frequency_mode, frequency, data_length, min, max)
        library.check_last_status_raise_on_error()

    def _get_burst_segment_count(self):
        """ Current burst segment count. """
        value = api.tiepie_hw_generator_get_burst_segment_count(self._handle)
        library.check_last_status_raise_on_error()
        return value

    def _set_burst_segment_count(self, value):
        api.tiepie_hw_generator_set_burst_segment_count(self._handle, value)
        library.check_last_status_raise_on_error()

    def verify_burst_segment_count(self, value):
        """ Verify if a burst segment count can be set, without actually setting the hardware itself.

        :param value: The requested burst segment count.
        :returns: The burst segment count that would have been set.
        .. version added:: 1.0
        """
        result = api.tiepie_hw_generator_verify_burst_segment_count(self._handle, value)
        library.check_last_status_raise_on_error()
        return result

    def verify_burst_segment_count_ex(self, value, generator_mode, signal_type, frequency_mode, frequency, data_length):
        """ Verify if a burst segment count mode, signal type, frequency mode, frequency and data length can be set, without actually setting the hardware.

        :param value: The requested burst segment count.
        :param generator_mode: The requested generator mode, a TIEPIE_HW_GM_* value.
        :param signal_type: The requested signal type, a TIEPIE_HW_ST_* value.
        :param frequency_mode: The requested frequency mode, a TIEPIE_HW_FM_* value. (Ignored for #TIEPIE_HW_ST_dC)
        :param frequency: The requested frequency in Hz.
        :param data_length: The requested data length in samples, only for #TIEPIE_HW_ST_ARBITRARY.
        :returns: The burst segment count that would have been set.
        .. version added:: 1.0
        """
        result = api.tiepie_hw_generator_verify_burst_segment_count_ex(self._handle, value, generator_mode, signal_type, frequency_mode, frequency, data_length)
        library.check_last_status_raise_on_error()
        return result

    connector_type = property(_get_connector_type)
    is_differential = property(_get_is_differential)
    impedance = property(_get_impedance)
    resolution = property(_get_resolution)
    output_value_min = property(_get_output_value_min)
    output_value_max = property(_get_output_value_max)
    is_controllable = property(_get_is_controllable)
    is_running = property(_get_is_running)
    status = property(_get_status)
    output_enable = property(_get_output_enable, _set_output_enable)
    has_output_invert = property(_get_has_output_invert)
    output_invert = property(_get_output_invert, _set_output_invert)
    signal_types = property(_get_signal_types)
    signal_type = property(_get_signal_type, _set_signal_type)
    has_amplitude = property(_get_has_amplitude)
    amplitude_min = property(_get_amplitude_min)
    amplitude_max = property(_get_amplitude_max)
    amplitude = property(_get_amplitude, _set_amplitude)
    amplitude_ranges = property(_get_amplitude_ranges)
    amplitude_range = property(_get_amplitude_range, _set_amplitude_range)
    amplitude_auto_ranging = property(_get_amplitude_auto_ranging, _set_amplitude_auto_ranging)
    has_offset = property(_get_has_offset)
    offset_min = property(_get_offset_min)
    offset_max = property(_get_offset_max)
    offset = property(_get_offset, _set_offset)
    frequency_modes = property(_get_frequency_modes)
    frequency_mode = property(_get_frequency_mode, _set_frequency_mode)
    has_frequency = property(_get_has_frequency)
    frequency_min = property(_get_frequency_min)
    frequency_max = property(_get_frequency_max)
    frequency = property(_get_frequency, _set_frequency)
    has_phase = property(_get_has_phase)
    phase_min = property(_get_phase_min)
    phase_max = property(_get_phase_max)
    phase = property(_get_phase, _set_phase)
    has_symmetry = property(_get_has_symmetry)
    symmetry_min = property(_get_symmetry_min)
    symmetry_max = property(_get_symmetry_max)
    symmetry = property(_get_symmetry, _set_symmetry)
    has_width = property(_get_has_width)
    width_min = property(_get_width_min)
    width_max = property(_get_width_max)
    width = property(_get_width, _set_width)
    has_edge_time = property(_get_has_edge_time)
    leading_edge_time_min = property(_get_leading_edge_time_min)
    leading_edge_time_max = property(_get_leading_edge_time_max)
    leading_edge_time = property(_get_leading_edge_time, _set_leading_edge_time)
    trailing_edge_time_min = property(_get_trailing_edge_time_min)
    trailing_edge_time_max = property(_get_trailing_edge_time_max)
    trailing_edge_time = property(_get_trailing_edge_time, _set_trailing_edge_time)
    has_data = property(_get_has_data)
    data_length_min = property(_get_data_length_min)
    data_length_max = property(_get_data_length_max)
    data_length = property(_get_data_length)
    data_raw_type = property(_get_data_raw_type)
    data_raw_value_min = property(_get_data_raw_value_min)
    data_raw_value_zero = property(_get_data_raw_value_zero)
    data_raw_value_max = property(_get_data_raw_value_max)
    modes = property(_get_modes)
    modes_native = property(_get_modes_native)
    mode = property(_get_mode, _set_mode)
    is_burst_active = property(_get_is_burst_active)
    burst_count_min = property(_get_burst_count_min)
    burst_count_max = property(_get_burst_count_max)
    burst_count = property(_get_burst_count, _set_burst_count)
    burst_sample_count_min = property(_get_burst_sample_count_min)
    burst_sample_count_max = property(_get_burst_sample_count_max)
    burst_sample_count = property(_get_burst_sample_count, _set_burst_sample_count)
    burst_segment_count_min = property(_get_burst_segment_count_min)
    burst_segment_count_max = property(_get_burst_segment_count_max)
    burst_segment_count = property(_get_burst_segment_count, _set_burst_segment_count)
