""" python-libtiepie - Python interface for libtiepie-hw library

Copyright (c) 2025 TiePie engineering

Website: https://www.tiepie.com/LibTiePie

"""

import datetime
import socket
import struct
from .const import *
from .types import *


def convert_date(value):
    """tiepie_hw_date to datetime.date object."""
    if value == 0:
        return None
    return datetime.date(value.year, value.month, value.day)


def convert_tristate(value):
    if value == TRISTATE_TRUE:
        return Tristate(True)
    elif value == TRISTATE_FALSE:
        return Tristate(False)
    else:
        return Tristate(None)


def auto_resolution_mode_str(value):
    result = []
    for k, v in AUTO_RESOLUTION_MODES.items():
        if (value & k) != 0:
            result.append(v)

    return ', '.join(result)


def coupling_str(value):
    result = []
    for k, v in COUPLINGS.items():
        if (value & k) != 0:
            result.append(v)

    return ', '.join(result)


def clock_output_str(value):
    result = []
    for k, v in CLOCK_OUTPUTS.items():
        if (value & k) != 0:
            result.append(v)

    return ', '.join(result)


def clock_source_str(value):
    result = []
    for k, v in CLOCK_SOURCES.items():
        if (value & k) != 0:
            result.append(v)

    return ', '.join(result)


def connector_type_str(value):
    result = []
    for k, v in CONNECTOR_TYPES.items():
        if (value & k) != 0:
            result.append(v)

    return ', '.join(result)


def device_type_str(value):
    result = []
    for k, v in DEVICE_TYPES.items():
        if (value & k) != 0:
            result.append(v)

    return ', '.join(result)


def frequency_mode_str(value):
    result = []
    for k, v in FREQUENCY_MODES.items():
        if (value & k) != 0:
            result.append(v)

    return ', '.join(result)


def generator_mode_str(value):
    result = []
    for k, v in GENERATOR_MODES.items():
        if (value & k) != 0:
            result.append(v)

    return ', '.join(result)


def generator_status_str(value):
    result = []
    for k, v in GENERATOR_STATUSS.items():
        if (value & k) != 0:
            result.append(v)

    return ', '.join(result)


def interface_str(value):
    result = []
    for k, v in INTERFACES.items():
        if (value & k) != 0:
            result.append(v)

    return ', '.join(result)


def measure_mode_str(value):
    result = []
    for k, v in MEASURE_MODES.items():
        if (value & k) != 0:
            result.append(v)

    return ', '.join(result)


def server_error_str(value):
    return SERVER_ERRORS[value]


def server_status_str(value):
    return SERVER_STATUSS[value]


def signal_type_str(value):
    result = []
    for k, v in SIGNAL_TYPES.items():
        if (value & k) != 0:
            result.append(v)

    return ', '.join(result)


def trigger_condition_str(value):
    result = []
    for k, v in TRIGGER_CONDITIONS.items():
        if (value & k) != 0:
            result.append(v)

    return ', '.join(result)


def trigger_kind_str(value):
    result = []
    for k, v in TRIGGER_KINDS.items():
        if (value & k) != 0:
            result.append(v)

    return ', '.join(result)


def trigger_level_mode_str(value):
    result = []
    for k, v in TRIGGER_LEVEL_MODES.items():
        if (value & k) != 0:
            result.append(v)

    return ', '.join(result)


def trigger_output_event_str(value):
    result = []
    for k, v in TRIGGER_OUTPUT_EVENTS.items():
        if (value & k) != 0:
            result.append(v)

    return ', '.join(result)
