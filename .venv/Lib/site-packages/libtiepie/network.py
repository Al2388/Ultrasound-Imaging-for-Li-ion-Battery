""" python-libtiepie - Python interface for libtiepie-hw library

Copyright (c) 2025 TiePie engineering

Website: https://www.tiepie.com/LibTiePie

"""

from ctypes import *
from .api import api
from .const import *
from .library import library
from .networkservers import NetworkServers


class Network(object):
    """"""

    def __init__(self):
        self._servers = NetworkServers()

    def _get_servers(self):
        return self._servers

    def _get_auto_detect_enabled(self):
        """  """
        value = api.tiepie_hw_network_get_auto_detect_enabled()
        library.check_last_status_raise_on_error()
        return value != BOOL_FALSE

    def _set_auto_detect_enabled(self, value):
        value = BOOL_TRUE if value else BOOL_FALSE
        api.tiepie_hw_network_set_auto_detect_enabled(value)
        library.check_last_status_raise_on_error()

    auto_detect_enabled = property(_get_auto_detect_enabled, _set_auto_detect_enabled)
    servers = property(_get_servers)


network = Network()
