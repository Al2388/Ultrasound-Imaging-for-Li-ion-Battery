""" python-libtiepie - Python interface for libtiepie-hw library

Copyright (c) 2025 TiePie engineering

Website: https://www.tiepie.com/LibTiePie

"""

from ctypes import *
from .api import api
from .const import *
from .utils import *
from .library import library
from .object import Object


class Server(Object):
    """"""

    def __init__(self, handle):
        super(Server, self).__init__(handle)

    def __eq__(self, other):
        if type(self) is type(other):
            return self._handle == other._handle
        return NotImplemented

    def __ne__(self, other):
        if type(self) is type(other):
            return not self.__eq__(other)
        return NotImplemented

    def connect(self, asynchronous=False):
        asynchronous = BOOL_TRUE if asynchronous else BOOL_FALSE
        result = api.tiepie_hw_server_connect(self._handle, asynchronous)
        library.check_last_status_raise_on_error()
        return result != BOOL_FALSE

    def disconnect(self, force=False):
        """ Disconnect from a specified network connected WiFiScope.

        :param force: If ``True`` all open devices are closed, if ``False`` remove only succeeds if no devices are open.
        :returns: ``True`` if successful, ``False`` otherwise.
        .. version added:: 1.0
        """
        force = BOOL_TRUE if force else BOOL_FALSE
        result = api.tiepie_hw_server_disconnect(self._handle, force)
        library.check_last_status_raise_on_error()
        return result != BOOL_FALSE

    def remove(self, force=False):
        """ Remove a specified network connected WiFiScope from the list of network connected WiFiScopes.

        :param force: If ``True`` all open devices are closed, if ``False`` remove only succeeds if no devices are open.
        :returns: ``True`` if successful, ``False`` otherwise.
        .. version added:: 1.0
        """
        force = BOOL_TRUE if force else BOOL_FALSE
        result = api.tiepie_hw_server_remove(self._handle, force)
        library.check_last_status_raise_on_error()
        return result != BOOL_FALSE

    def _get_status(self):
        """ Retrieve the status of a specified network connected WiFiScope """
        value = api.tiepie_hw_server_get_status(self._handle)
        library.check_last_status_raise_on_error()
        return value

    def _get_last_error(self):
        """ Last error from a specified network connected WiFiScope """
        value = api.tiepie_hw_server_get_last_error(self._handle)
        library.check_last_status_raise_on_error()
        return value

    def _get_url(self):
        """ URL of the specified network connected WiFiScope. """
        length = api.tiepie_hw_server_get_url(self._handle, None, 0)
        library.check_last_status_raise_on_error()
        buf = create_string_buffer(length + 1)
        api.tiepie_hw_server_get_url(self._handle, buf, length + 1)
        library.check_last_status_raise_on_error()
        return buf.value.decode('utf-8')

    def _get_id(self):
        """ Id of the specified network connected WiFiScope. """
        length = api.tiepie_hw_server_get_id(self._handle, None, 0)
        library.check_last_status_raise_on_error()
        buf = create_string_buffer(length + 1)
        api.tiepie_hw_server_get_id(self._handle, buf, length + 1)
        library.check_last_status_raise_on_error()
        return buf.value.decode('utf-8')

    def _get_ip_address(self):
        """ IP address of the specified network connected WiFiScope. """
        length = api.tiepie_hw_server_get_ip_address(self._handle, None, 0)
        library.check_last_status_raise_on_error()
        buf = create_string_buffer(length + 1)
        api.tiepie_hw_server_get_ip_address(self._handle, buf, length + 1)
        library.check_last_status_raise_on_error()
        return buf.value.decode('utf-8')

    def _get_ip_port(self):
        """ IP port number of the specified network connected WiFiScope. """
        value = api.tiepie_hw_server_get_ip_port(self._handle)
        library.check_last_status_raise_on_error()
        return value

    def _get_name(self):
        """ Name of the specified network connected WiFiScope. """
        length = api.tiepie_hw_server_get_name(self._handle, None, 0)
        library.check_last_status_raise_on_error()
        buf = create_string_buffer(length + 1)
        api.tiepie_hw_server_get_name(self._handle, buf, length + 1)
        library.check_last_status_raise_on_error()
        return buf.value.decode('utf-8')

    def _get_description(self):
        """ Description of the specified network connected WiFiScope. """
        length = api.tiepie_hw_server_get_description(self._handle, None, 0)
        library.check_last_status_raise_on_error()
        buf = create_string_buffer(length + 1)
        api.tiepie_hw_server_get_description(self._handle, buf, length + 1)
        library.check_last_status_raise_on_error()
        return buf.value.decode('utf-8')

    def _get_version(self):
        """ Software version number of the specified network connected WiFiScope. """
        length = api.tiepie_hw_server_get_version(self._handle, None, 0)
        library.check_last_status_raise_on_error()
        buf = create_string_buffer(length + 1)
        api.tiepie_hw_server_get_version(self._handle, buf, length + 1)
        library.check_last_status_raise_on_error()
        return buf.value.decode('utf-8')

    status = property(_get_status)
    last_error = property(_get_last_error)
    url = property(_get_url)
    id = property(_get_id)
    ip_address = property(_get_ip_address)
    ip_port = property(_get_ip_port)
    name = property(_get_name)
    description = property(_get_description)
    version = property(_get_version)
