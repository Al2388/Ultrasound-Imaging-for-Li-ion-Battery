""" python-libtiepie - Python interface for libtiepie-hw library

Copyright (c) 2025 TiePie engineering

Website: https://www.tiepie.com/LibTiePie

"""


class ObjectList(object):
    """"""

    def __init__(self):
        self._items = []

    def __getitem__(self, key):
        if isinstance(key, slice):
            return self._items.__getitem__(key)
        return self._items[key]

    def __len__(self):
        return len(self._items)

    def _get_count(self):
        return len(self._items)

    count = property(_get_count)
