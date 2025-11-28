""" python-libtiepie - Python interface for libtiepie-hw library

Copyright (c) 2025 TiePie engineering

Website: https://www.tiepie.com/LibTiePie

"""

from collections import namedtuple
from ctypes import *


tiepie_hw_devicelist_callback = CFUNCTYPE(None, c_void_p, c_uint32, c_uint32)
tiepie_hw_handle_callback = CFUNCTYPE(None, c_void_p, c_uint32)
tiepie_hw_event_callback = CFUNCTYPE(None, c_void_p, c_int32, c_uint32)


class tiepie_hw_date(Structure):
    _fields_ = [
        ("year", c_uint16),
        ("month", c_uint8),
        ("day", c_uint8)]


class tiepie_hw_version(Structure):
    _fields_ = [
        ("major", c_uint16),
        ("minor", c_uint16),
        ("patch", c_uint16),
        ("build", c_uint16),
        ("extra", c_char_p)]

    def __str__(self):
        return f'{self.major}.{self.minor}.{self.patch}.{self.build}{"" if self.extra is None else self.extra.decode("utf-8")}'


class Tristate(object):  # See: http://stackoverflow.com/a/9504358
    def __init__(self, value=None):
        if any(value is v for v in (True, False, None)):
            self.value = value
        else:
            raise ValueError('Tristate value must be True, False, or None')

    def __eq__(self, other):
        return self.value is other

    def __ne__(self, other):
        return self.value is not other

    def __nonzero__(self):
        raise TypeError('Tristate value may not be used as implicit Boolean')

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return f'Tristate({self.value})'
