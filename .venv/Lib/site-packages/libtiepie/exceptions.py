""" python-libtiepie - Python interface for libtiepie-hw library

Copyright (c) 2025 TiePie engineering

Website: https://www.tiepie.com/LibTiePie

"""


class LibTiePieException(Exception):
    """The base class for all LibTiePie exceptions."""
    def __init__(self, status, message):
        super(LibTiePieException, self).__init__(message)
        self.status = status


class UnsuccessfulError(LibTiePieException):
    """"""

    def __init__(self):
        super(UnsuccessfulError, self).__init__(UnsuccessfulError, 'Unsuccessful')


class NotSupportedError(LibTiePieException):
    """"""

    def __init__(self):
        super(NotSupportedError, self).__init__(NotSupportedError, 'Not supported')


class InvalidHandleError(LibTiePieException):
    """"""

    def __init__(self):
        super(InvalidHandleError, self).__init__(InvalidHandleError, 'Invalid handle')


class InvalidValueError(LibTiePieException):
    """"""

    def __init__(self):
        super(InvalidValueError, self).__init__(InvalidValueError, 'Invalid value')


class InvalidChannelError(LibTiePieException):
    """"""

    def __init__(self):
        super(InvalidChannelError, self).__init__(InvalidChannelError, 'Invalid channel')


class InvalidTriggerSourceError(LibTiePieException):
    """"""

    def __init__(self):
        super(InvalidTriggerSourceError, self).__init__(InvalidTriggerSourceError, 'Invalid trigger source')


class InvalidDeviceTypeError(LibTiePieException):
    """"""

    def __init__(self):
        super(InvalidDeviceTypeError, self).__init__(InvalidDeviceTypeError, 'Invalid device type')


class InvalidDeviceIndexError(LibTiePieException):
    """"""

    def __init__(self):
        super(InvalidDeviceIndexError, self).__init__(InvalidDeviceIndexError, 'Invalid device index')


class InvalidProductIDError(LibTiePieException):
    """"""

    def __init__(self):
        super(InvalidProductIDError, self).__init__(InvalidProductIDError, 'Invalid product id')


class InvalidDeviceSerialNumberError(LibTiePieException):
    """"""

    def __init__(self):
        super(InvalidDeviceSerialNumberError, self).__init__(InvalidDeviceSerialNumberError, 'Invalid device serialnumber')


class ObjectGoneError(LibTiePieException):
    """"""

    def __init__(self):
        super(ObjectGoneError, self).__init__(ObjectGoneError, 'Object gone')


class InternalAddressError(LibTiePieException):
    """"""

    def __init__(self):
        super(InternalAddressError, self).__init__(InternalAddressError, 'Internal address')


class NotControllableError(LibTiePieException):
    """"""

    def __init__(self):
        super(NotControllableError, self).__init__(NotControllableError, 'Not controllable')


class BitError(LibTiePieException):
    """"""

    def __init__(self):
        super(BitError, self).__init__(BitError, 'Bit error')


class NoAcknowledgeError(LibTiePieException):
    """"""

    def __init__(self):
        super(NoAcknowledgeError, self).__init__(NoAcknowledgeError, 'No acknowledge')


class InvalidContainedDeviceSerialNumberError(LibTiePieException):
    """"""

    def __init__(self):
        super(InvalidContainedDeviceSerialNumberError, self).__init__(InvalidContainedDeviceSerialNumberError, 'Invalid contained device serialnumber')


class InvalidInputError(LibTiePieException):
    """"""

    def __init__(self):
        super(InvalidInputError, self).__init__(InvalidInputError, 'Invalid input')


class InvalidOutputError(LibTiePieException):
    """"""

    def __init__(self):
        super(InvalidOutputError, self).__init__(InvalidOutputError, 'Invalid output')


class NotAvailableError(LibTiePieException):
    """"""

    def __init__(self):
        super(NotAvailableError, self).__init__(NotAvailableError, 'Not available')


class InvalidFirmwareError(LibTiePieException):
    """"""

    def __init__(self):
        super(InvalidFirmwareError, self).__init__(InvalidFirmwareError, 'Invalid firmware')


class InvalidIndexError(LibTiePieException):
    """"""

    def __init__(self):
        super(InvalidIndexError, self).__init__(InvalidIndexError, 'Invalid index')


class InvalidEepromError(LibTiePieException):
    """"""

    def __init__(self):
        super(InvalidEepromError, self).__init__(InvalidEepromError, 'Invalid eeprom')


class InitializationFailedError(LibTiePieException):
    """"""

    def __init__(self):
        super(InitializationFailedError, self).__init__(InitializationFailedError, 'Initialization failed')


class LibraryNotInitializedError(LibTiePieException):
    """"""

    def __init__(self):
        super(LibraryNotInitializedError, self).__init__(LibraryNotInitializedError, 'Library not initialized')


class NoTriggerEnabledError(LibTiePieException):
    """"""

    def __init__(self):
        super(NoTriggerEnabledError, self).__init__(NoTriggerEnabledError, 'No trigger enabled')


class SynchronizationFailedError(LibTiePieException):
    """"""

    def __init__(self):
        super(SynchronizationFailedError, self).__init__(SynchronizationFailedError, 'Synchronization failed')


class InvalidHS56CombinedDeviceError(LibTiePieException):
    """"""

    def __init__(self):
        super(InvalidHS56CombinedDeviceError, self).__init__(InvalidHS56CombinedDeviceError, 'Invalid hs56 combined device')


class MeasurementRunningError(LibTiePieException):
    """"""

    def __init__(self):
        super(MeasurementRunningError, self).__init__(MeasurementRunningError, 'Measurement running')


class WirelesstriggermodulenotconnectedError(LibTiePieException):
    """"""

    def __init__(self):
        super(WirelesstriggermodulenotconnectedError, self).__init__(WirelesstriggermodulenotconnectedError, 'Wirelesstriggermodulenotconnected')


class InitializationError10001Error(LibTiePieException):
    """"""

    def __init__(self):
        super(InitializationError10001Error, self).__init__(InitializationError10001Error, 'Initialization error 10001')


class InitializationError10002Error(LibTiePieException):
    """"""

    def __init__(self):
        super(InitializationError10002Error, self).__init__(InitializationError10002Error, 'Initialization error 10002')


class InitializationError10003Error(LibTiePieException):
    """"""

    def __init__(self):
        super(InitializationError10003Error, self).__init__(InitializationError10003Error, 'Initialization error 10003')


class InitializationError10004Error(LibTiePieException):
    """"""

    def __init__(self):
        super(InitializationError10004Error, self).__init__(InitializationError10004Error, 'Initialization error 10004')


class InitializationError10005Error(LibTiePieException):
    """"""

    def __init__(self):
        super(InitializationError10005Error, self).__init__(InitializationError10005Error, 'Initialization error 10005')


class InitializationError10006Error(LibTiePieException):
    """"""

    def __init__(self):
        super(InitializationError10006Error, self).__init__(InitializationError10006Error, 'Initialization error 10006')


class InitializationError10007Error(LibTiePieException):
    """"""

    def __init__(self):
        super(InitializationError10007Error, self).__init__(InitializationError10007Error, 'Initialization error 10007')
