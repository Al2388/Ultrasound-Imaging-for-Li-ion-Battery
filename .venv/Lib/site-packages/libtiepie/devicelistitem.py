""" python-libtiepie - Python interface for libtiepie-hw library

Copyright (c) 2025 TiePie engineering

Website: https://www.tiepie.com/LibTiePie

"""

from array import array
from ctypes import create_string_buffer
from .api import api
from .const import *
from .utils import *
from .library import library
from .object import Object
from .oscilloscope import Oscilloscope
from .generator import Generator
from .server import Server


class DeviceListItem(Object):
    """"""

    def __init__(self, handle):
        super(DeviceListItem, self).__init__(handle)

    def open_device(self, device_type):
        """ Open a device .

        :param device_type: A device type.
        :returns: Instance of :class:`.Oscilloscope` or :class:`.Generator`.
        .. version added:: 1.0
        """
        result = api.tiepie_hw_devicelistitem_open_device(self._handle, device_type)
        library.check_last_status_raise_on_error()
        return library.create_object(result)

    def open_oscilloscope(self):
        """ Open an oscilloscope .

        :returns: Instance of :class:`.Oscilloscope`.
        .. version added:: 1.0
        """
        result = api.tiepie_hw_devicelistitem_open_oscilloscope(self._handle)
        library.check_last_status_raise_on_error()
        return Oscilloscope(result)

    def open_generator(self):
        """ Open a generator .

        :returns: Instance of :class:`.Generator`.
        .. version added:: 1.0
        """
        result = api.tiepie_hw_devicelistitem_open_generator(self._handle)
        library.check_last_status_raise_on_error()
        return Generator(result)

    def _get_is_demo(self):
        """  """
        value = api.tiepie_hw_devicelistitem_is_demo(self._handle)
        library.check_last_status_raise_on_error()
        return value != BOOL_FALSE

    def can_open(self, device_type):
        """ Check whether the listed device can be opened.

        :param device_type: A device type.
        :returns: ``True`` if the device can be opened or ``False`` if not.
        .. version added:: 1.0
        """
        result = api.tiepie_hw_devicelistitem_can_open(self._handle, device_type)
        library.check_last_status_raise_on_error()
        return result != BOOL_FALSE

    def opened_by(self, device_type):
        length = api.tiepie_hw_devicelistitem_opened_by(self._handle, device_type, 'None', '0')
        library.check_last_status_raise_on_error()
        buf = create_string_buffer(length + 1)
        api.tiepie_hw_devicelistitem_opened_by(self._handle, device_type, buf, length + 1)
        library.check_last_status_raise_on_error()
        return buf.value.decode('utf-8')

    def _get_product_id(self):
        """ Product id. """
        value = api.tiepie_hw_devicelistitem_get_product_id(self._handle)
        library.check_last_status_raise_on_error()
        return value

    def _get_name(self):
        """ Full name. """
        length = api.tiepie_hw_devicelistitem_get_name(self._handle, None, 0)
        library.check_last_status_raise_on_error()
        buf = create_string_buffer(length + 1)
        api.tiepie_hw_devicelistitem_get_name(self._handle, buf, length + 1)
        library.check_last_status_raise_on_error()
        return buf.value.decode('utf-8')

    def _get_name_short(self):
        """ Short name. """
        length = api.tiepie_hw_devicelistitem_get_name_short(self._handle, None, 0)
        library.check_last_status_raise_on_error()
        buf = create_string_buffer(length + 1)
        api.tiepie_hw_devicelistitem_get_name_short(self._handle, buf, length + 1)
        library.check_last_status_raise_on_error()
        return buf.value.decode('utf-8')

    def _get_name_shortest(self):
        """ Short name wihout model postfix. """
        length = api.tiepie_hw_devicelistitem_get_name_shortest(self._handle, None, 0)
        library.check_last_status_raise_on_error()
        buf = create_string_buffer(length + 1)
        api.tiepie_hw_devicelistitem_get_name_shortest(self._handle, buf, length + 1)
        library.check_last_status_raise_on_error()
        return buf.value.decode('utf-8')

    def _get_calibration_date(self):
        """ Calibration date. """
        value = api.tiepie_hw_devicelistitem_get_calibration_date(self._handle)
        library.check_last_status_raise_on_error()
        return convert_date(value)

    def _get_serial_number(self):
        """ Serial number. """
        value = api.tiepie_hw_devicelistitem_get_serial_number(self._handle)
        library.check_last_status_raise_on_error()
        return value

    def _get_ip_address(self):
        """ IP address. """
        length = api.tiepie_hw_devicelistitem_get_ip_address(self._handle, None, 0)
        library.check_last_status_raise_on_error()
        buf = create_string_buffer(length + 1)
        api.tiepie_hw_devicelistitem_get_ip_address(self._handle, buf, length + 1)
        library.check_last_status_raise_on_error()
        return buf.value.decode('utf-8')

    def _get_ip_port(self):
        """ IP port number. """
        value = api.tiepie_hw_devicelistitem_get_ip_port(self._handle)
        library.check_last_status_raise_on_error()
        return value

    def _get_has_server(self):
        """ Check whether the listed device is connected to a server. """
        value = api.tiepie_hw_devicelistitem_has_server(self._handle)
        library.check_last_status_raise_on_error()
        return value != BOOL_FALSE

    def _get_server(self):
        """ Server handle of the server the listed device is connected to. """
        value = api.tiepie_hw_devicelistitem_get_server(self._handle)
        library.check_last_status_raise_on_error()
        return Server(value)

    def _get_types(self):
        """ Device types. """
        value = api.tiepie_hw_devicelistitem_get_types(self._handle)
        library.check_last_status_raise_on_error()
        return value

    def _get_contained_serial_numbers(self):
        """  """
        count = api.tiepie_hw_devicelistitem_get_contained_serial_numbers(self._handle, None, 0)
        library.check_last_status_raise_on_error()
        values = (c_uint32 * count)()
        api.tiepie_hw_devicelistitem_get_contained_serial_numbers(self._handle, values, count)
        library.check_last_status_raise_on_error()
        return array('L', values)

    is_demo = property(_get_is_demo)
    product_id = property(_get_product_id)
    name = property(_get_name)
    name_short = property(_get_name_short)
    name_shortest = property(_get_name_shortest)
    calibration_date = property(_get_calibration_date)
    serial_number = property(_get_serial_number)
    ip_address = property(_get_ip_address)
    ip_port = property(_get_ip_port)
    has_server = property(_get_has_server)
    server = property(_get_server)
    types = property(_get_types)
    contained_serial_numbers = property(_get_contained_serial_numbers)
