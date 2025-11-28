""" python-libtiepie - Python interface for libtiepie-hw library

Copyright (c) 2025 TiePie engineering

Website: https://www.tiepie.com/LibTiePie

"""

from .api import api
from .const import *
from .library import library


class Object(object):
    """"""

    def __init__(self, handle):
        self._handle = handle

    def __del__(self):
        api.tiepie_hw_object_close(self._handle)

    def _get_is_removed(self):
        """ Check whether an object is removed. """
        value = api.tiepie_hw_object_is_removed(self._handle)
        library.check_last_status_raise_on_error()
        return value != BOOL_FALSE

    def _get_interfaces(self):
        """ Check which interfaces are supported by the specified object. """
        value = api.tiepie_hw_object_get_interfaces(self._handle)
        library.check_last_status_raise_on_error()
        return value

    is_removed = property(_get_is_removed)
    interfaces = property(_get_interfaces)
