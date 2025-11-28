""" python-libtiepie - Python interface for libtiepie-hw library

Copyright (c) 2025 TiePie engineering

Website: https://www.tiepie.com/LibTiePie

"""

from array import array
from .api import api
from .utils import *
from .const import *
from .exceptions import *


class Library(object):
    """"""

    def __init__(self):
        self.__exit = api.tiepie_hw_fini  # Prevent garbage collector from freeing the exit function
        api.tiepie_hw_init()

    def __del__(self):
        self.__exit()

    def _get_config_str(self):
        result = '0x'
        for v in self.config:
            result += '{0:02x}'.format(v)
        return result

    def check_last_status_raise_on_error(self):
        status = self.last_status
        if status >= STATUS_SUCCESS:
            return
        elif status == STATUS_UNSUCCESSFUL:
            raise UnsuccessfulError()
        elif status == STATUS_NOT_SUPPORTED:
            raise NotSupportedError()
        elif status == STATUS_INVALID_HANDLE:
            raise InvalidHandleError()
        elif status == STATUS_INVALID_VALUE:
            raise InvalidValueError()
        elif status == STATUS_INVALID_CHANNEL:
            raise InvalidChannelError()
        elif status == STATUS_INVALID_TRIGGER_SOURCE:
            raise InvalidTriggerSourceError()
        elif status == STATUS_INVALID_DEVICE_TYPE:
            raise InvalidDeviceTypeError()
        elif status == STATUS_INVALID_DEVICE_INDEX:
            raise InvalidDeviceIndexError()
        elif status == STATUS_INVALID_PRODUCT_ID:
            raise InvalidProductIDError()
        elif status == STATUS_INVALID_DEVICE_SERIALNUMBER:
            raise InvalidDeviceSerialNumberError()
        elif status == STATUS_OBJECT_GONE:
            raise ObjectGoneError()
        elif status == STATUS_INTERNAL_ADDRESS:
            raise InternalAddressError()
        elif status == STATUS_NOT_CONTROLLABLE:
            raise NotControllableError()
        elif status == STATUS_BIT_ERROR:
            raise BitError()
        elif status == STATUS_NO_ACKNOWLEDGE:
            raise NoAcknowledgeError()
        elif status == STATUS_INVALID_CONTAINED_DEVICE_SERIALNUMBER:
            raise InvalidContainedDeviceSerialNumberError()
        elif status == STATUS_INVALID_INPUT:
            raise InvalidInputError()
        elif status == STATUS_INVALID_OUTPUT:
            raise InvalidOutputError()
        elif status == STATUS_NOT_AVAILABLE:
            raise NotAvailableError()
        elif status == STATUS_INVALID_FIRMWARE:
            raise InvalidFirmwareError()
        elif status == STATUS_INVALID_INDEX:
            raise InvalidIndexError()
        elif status == STATUS_INVALID_EEPROM:
            raise InvalidEepromError()
        elif status == STATUS_INITIALIZATION_FAILED:
            raise InitializationFailedError()
        elif status == STATUS_LIBRARY_NOT_INITIALIZED:
            raise LibraryNotInitializedError()
        elif status == STATUS_NO_TRIGGER_ENABLED:
            raise NoTriggerEnabledError()
        elif status == STATUS_SYNCHRONIZATION_FAILED:
            raise SynchronizationFailedError()
        elif status == STATUS_INVALID_HS56_COMBINED_DEVICE:
            raise InvalidHS56CombinedDeviceError()
        elif status == STATUS_MEASUREMENT_RUNNING:
            raise MeasurementRunningError()
        elif status == STATUS_WIRELESSTRIGGERMODULENOTCONNECTED:
            raise WirelesstriggermodulenotconnectedError()
        elif status == STATUS_INITIALIZATION_ERROR_10001:
            raise InitializationError10001Error()
        elif status == STATUS_INITIALIZATION_ERROR_10002:
            raise InitializationError10002Error()
        elif status == STATUS_INITIALIZATION_ERROR_10003:
            raise InitializationError10003Error()
        elif status == STATUS_INITIALIZATION_ERROR_10004:
            raise InitializationError10004Error()
        elif status == STATUS_INITIALIZATION_ERROR_10005:
            raise InitializationError10005Error()
        elif status == STATUS_INITIALIZATION_ERROR_10006:
            raise InitializationError10006Error()
        elif status == STATUS_INITIALIZATION_ERROR_10007:
            raise InitializationError10007Error()
        else:
            raise LibTiePieException(status, self.last_status_str)

    def create_object(self, handle):
        interfaces = api.tiepie_hw_object_get_interfaces(handle)
        self.check_last_status_raise_on_error()

        if interfaces == (INTERFACE_DEVICE | INTERFACE_OSCILLOSCOPE):
            from .oscilloscope import Oscilloscope
            return Oscilloscope(handle)
        elif interfaces == (INTERFACE_DEVICE | INTERFACE_GENERATOR):
            from .generator import Generator
            return Generator(handle)
        elif interfaces == INTERFACE_DEVICE:
            from .device import Device
            return Device(handle)
        elif interfaces == INTERFACE_SERVER:
            from .server import Server
            return Server(handle)
        else:
            raise InvalidValueError()

    def _get_is_initialized(self):
        """ Check whether the library's internal resources are initialized. """
        value = api.tiepie_hw_is_initialized()
        return value != BOOL_FALSE

    def fini(self):
        api.tiepie_hw_fini()

    def _get_version(self):
        """ Get library version info. """
        return api.tiepie_hw_get_version().contents

    def _get_config(self):
        """ Library configuration number. """
        count = api.tiepie_hw_get_config(None, 0)
        values = (c_uint8 * count)()
        api.tiepie_hw_get_config(values, count)
        return array('B', values)

    def _get_last_status(self):
        """ Last status value. """
        return api.tiepie_hw_get_last_status()

    def _get_last_status_str(self):
        """ Last status value as text. """
        value = api.tiepie_hw_get_last_status_str()
        return value.decode('utf-8')

    is_initialized = property(_get_is_initialized)
    version = property(_get_version)
    config = property(_get_config)
    last_status = property(_get_last_status)
    last_status_str = property(_get_last_status_str)
    config_str = property(_get_config_str)


library = Library()
