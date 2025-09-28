"""
Core data types for NoisePy DAS
"""

import sys
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import obspy

INVALID_COORD = -sys.float_info.max


@dataclass
class ChannelType:
    """
    A type of channel, e.g. XXZ, but not associated with a particular station
    """

    name: str
    location: str = ""

    def __post_init__(self):
        # Handle location embedded in name with underscore
        if "_" in self.name:
            parts = self.name.split("_")
            self.name = parts[0]
            self.location = parts[1]

        # Japanese channels use 'U' (up) for the vertical direction. Here we normalize to 'Z'
        if self.name[-1] == "U":
            self.name = self.name.replace("U", "Z")

    def __repr__(self) -> str:
        if len(self.location) > 0:
            return f"{self.name}_{self.location}"
        else:
            return self.name

    def get_orientation(self) -> str:
        if "_" in self.name:
            return self.name.split("_")[0][-1]
        else:
            return self.name[-1]


@dataclass
class Station:
    """
    A seismic station with network, name, and optional coordinates
    """
    network: str  # 2 chars
    name: str     # 3-5 chars
    lat: float
    lon: float
    elevation: float
    location: str

    def __init__(
        self,
        network: str,
        name: str,
        lat: float = INVALID_COORD,
        lon: float = INVALID_COORD,
        elevation: float = INVALID_COORD,
        location: str = "",
    ):
        self.network = network
        self.name = name
        self.lat = lat
        self.lon = lon
        self.elevation = elevation
        self.location = location

    @staticmethod
    def parse(sta: str):
        """Parse from: CI.ARV format"""
        parts = sta.split(".")
        if len(parts) != 2:
            return None
        return Station(parts[0], parts[1])

    def valid(self) -> bool:
        """Check if station has valid coordinates"""
        return min(self.lat, self.lon, self.elevation) > INVALID_COORD

    def __repr__(self) -> str:
        return f"{self.network}.{self.name}"

    def __hash__(self) -> int:
        return str(self).__hash__()

    def __eq__(self, __value: object) -> bool:
        return str(self) == str(__value)


@dataclass
class Channel:
    """
    A channel instance belonging to a station. E.g. CI.ARV.XXZ
    """

    type: ChannelType
    station: Station

    def __repr__(self) -> str:
        return f"{self.station}.{self.type}"


class ChannelData:
    """
    A 1D time series of channel data

    Attributes:
        stream: obspy Stream object containing the data
        data: numpy array of the time series values  
        sampling_rate: sampling rate in Hz
        start_timestamp: start time as seconds since epoch
    """

    def __init__(self, stream: obspy.Stream):
        self.stream = stream
        if len(stream) > 0:
            self.data = stream[0].data[:]
            self.sampling_rate = stream[0].stats.sampling_rate
            self.start_timestamp = stream[0].stats.starttime.timestamp
        else:
            self.data = np.empty(0)
            self.sampling_rate = 1.0
            self.start_timestamp = 0.0

    @staticmethod
    def empty():
        """Create empty ChannelData instance"""
        return ChannelData(obspy.Stream([obspy.Trace(np.empty(0))]))