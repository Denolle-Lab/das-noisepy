"""
Base classes for data stores
"""

from abc import ABC, abstractmethod
from typing import List

import obspy
from datetimerange import DateTimeRange

from .datatypes import Channel, ChannelData, Station


class DataStore(ABC):
    """
    A base abstraction over a data source for seismic data
    """

    @abstractmethod
    def get_channels(self, timespan: DateTimeRange) -> List[Channel]:
        """Get list of channels available for the given timespan"""
        pass

    @abstractmethod
    def get_timespans(self) -> List[DateTimeRange]:
        """Get list of all available timespans"""
        pass


class RawDataStore(DataStore):
    """
    A class for reading raw data for a given channel.
    """

    @abstractmethod
    def read_data(self, timespan: DateTimeRange, chan: Channel) -> ChannelData:
        """Read data for a specific channel and timespan"""
        pass

    @abstractmethod
    def get_inventory(self, timespan: DateTimeRange, station: Station) -> obspy.Inventory:
        """Get station inventory information"""
        pass