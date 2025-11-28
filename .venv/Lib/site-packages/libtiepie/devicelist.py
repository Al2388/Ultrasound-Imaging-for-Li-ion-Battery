""" python-libtiepie - Python interface for libtiepie-hw library

Copyright (c) 2025 TiePie engineering

Website: https://www.tiepie.com/LibTiePie

"""

from ctypes import c_uint32
from .api import api
from .const import *
from .library import library
from .devicelistitem import DeviceListItem
from .exceptions import InvalidDeviceIndexError


class DeviceList(object):
    """"""

    def __getitem__(self, index):
        try:
            return self.get_item_by_index(index)
        except (InvalidDeviceIndexError):
            raise IndexError()

    def __len__(self):
        return self.count

    def get_item_by_product_id(self, pid):
        handle = api.tiepie_hw_devicelist_get_item_by_product_id(pid)
        library.check_last_status_raise_on_error()
        return DeviceListItem(handle)

    def get_item_by_index(self, index):
        handle = api.tiepie_hw_devicelist_get_item_by_index(index)
        library.check_last_status_raise_on_error()
        return DeviceListItem(handle)

    def get_item_by_serial_number(self, serial_number):
        handle = api.tiepie_hw_devicelist_get_item_by_serial_number(serial_number)
        library.check_last_status_raise_on_error()
        return DeviceListItem(handle)

    def update(self):
        api.tiepie_hw_devicelist_update()
        library.check_last_status_raise_on_error()

    def _get_usb_hotplug_detect_enabled(self):
        """ Current enabled state of the USB hot plug detection. """
        value = api.tiepie_hw_devicelist_get_usb_hotplug_detect_enabled()
        library.check_last_status_raise_on_error()
        return value != BOOL_FALSE

    def _set_usb_hotplug_detect_enabled(self, value):
        value = BOOL_TRUE if value else BOOL_FALSE
        api.tiepie_hw_devicelist_set_usb_hotplug_detect_enabled(value)
        library.check_last_status_raise_on_error()

    def _get_count(self):
        """ Number of devices in the device list. """
        value = api.tiepie_hw_devicelist_get_count()
        library.check_last_status_raise_on_error()
        return value

    def _get_demo_device_info(self):
        """  """
        value = api.tiepie_hw_devicelist_get_demo_device_info()
        library.check_last_status_raise_on_error()
        return value

    def create_demo_device(self, product_id):
        """ Create a demo instrument.

        :param product_id: The product ID of the demo instrument to create
        :returns: Serial number of the demo device, or zero on error.
        .. version added:: 1.0
        """
        result = api.tiepie_hw_devicelist_create_demo_device(product_id)
        library.check_last_status_raise_on_error()
        return result

    def create_combined_device(self, devices):
        """ Create a combined instrument.

        :param device: :class:`list` of Device instances.
        :returns: Device list item of combined device.
        .. version added:: 1.0
        """
        handles = (c_uint32 * len(devices))()
        i = 0
        for device in devices:
            handles[i] = device._handle
            i += 1
        serial_number = api.tiepie_hw_devicelist_create_combined_device(handles, len(handles))
        library.check_last_status_raise_on_error()

        handle = api.tiepie_hw_devicelist_get_item_by_serial_number(serial_number)
        library.check_last_status_raise_on_error()

        return DeviceListItem(handle)

    def create_and_open_combined_device(self, devices):
        """ Create and open a combined instrument.

        :param device: :class:`list` of Device instances.
        :returns: Instance of combined device.
        .. version added:: 1.0
        """
        handles = (c_uint32 * len(devices))()
        i = 0
        for device in devices:
            handles[i] = device._handle
            i += 1
        handle = api.tiepie_hw_devicelist_create_and_open_combined_device(handles, len(handles))
        library.check_last_status_raise_on_error()

        return library.create_object(handle)

    def remove_device(self, serial_number, force=False):
        """ Remove an instrument from the device list so it can be used by other applications.

        :param serial_number: Serial number of the device to remove.
        :param force: Force the removal, even when the device is open.
        .. version added:: 1.0
        """
        force = BOOL_TRUE if force else BOOL_FALSE
        api.tiepie_hw_devicelist_remove_device(serial_number, force)
        library.check_last_status_raise_on_error()

    def remove_unused_devices(self):
        api.tiepie_hw_devicelist_remove_unused_devices()
        library.check_last_status_raise_on_error()

    usb_hotplug_detect_enabled = property(_get_usb_hotplug_detect_enabled, _set_usb_hotplug_detect_enabled)
    count = property(_get_count)
    demo_device_info = property(_get_demo_device_info)


device_list = DeviceList()
