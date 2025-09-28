"""
HDF5 data store for DAS (Distributed Acoustic Sensing) data
"""

import logging
import os
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import List

import h5py
import obspy
from datetimerange import DateTimeRange

from .datatypes import Channel, ChannelData, ChannelType, Station
from .stores import RawDataStore
from .utils import TimeLogger, fs_join, get_filesystem

logger = logging.getLogger(__name__)


class DASH5DataStore(RawDataStore):
    """
    A data store implementation to read from a directory of HDF5 (.h5) files.
    Every file is a minute recording and each .h5 file contains the data for all channels.
    
    This class is designed to work with DAS data stored in HDF5 format where:
    - Each file contains 1 minute of data
    - Data is stored in /Acquisition/Raw[0]/RawData dataset
    - Timestamps are stored in /Acquisition/Raw[0]/RawDataTime dataset
    """

    def __init__(
        self,
        path: str,
        sampling_rate: int,
        channel_numbers: List[int],
        file_naming: str = "%Y-%m-%d-%H-%M-%S.h5",
        array_name: str = "DAS",
        date_range: DateTimeRange = None,
        storage_options: dict = None,
    ):
        """
        Parameters:
            path: path to look for h5 files. Must be a local file directory
            sampling_rate: sampling rate of the data in Hz
            file_naming: a string format to parse the file name. Must contain a datetime format
            channel_numbers: a list of channel numbers to read
            array_name: name of the array (used as network name)
            date_range: optional date range to limit data loading
            storage_options: optional fsspec storage options
        """
        super().__init__()
        if storage_options is None:
            storage_options = {}
        
        self.fs = get_filesystem(path, storage_options=storage_options)
        self.sampling_rate = sampling_rate
        self.array_name = array_name
        self.channel_numbers = channel_numbers
        self.file_naming = file_naming
        self.path = os.path.abspath(path)
        self.paths = {}
        self.channels = defaultdict(list)
        
        # Handle timezone-aware date range
        if date_range is not None and date_range.start_datetime.tzinfo is None:
            start_datetime = date_range.start_datetime.replace(tzinfo=timezone.utc)
            end_datetime = date_range.end_datetime.replace(tzinfo=timezone.utc)
            date_range = DateTimeRange(start_datetime, end_datetime)

        self.date_range = date_range

        # Load available files if no date range is specified
        if date_range is None:
            for file in self.fs.glob(fs_join(self.path, "*.h5")):
                self._load_channels(file)

    def _load_channels(self, full_path: str):
        """Load channel information from a file path"""
        tlog = TimeLogger(logger=logger, level=logging.INFO)
        msfiles = [f for f in self.fs.glob(full_path)]
        tlog.log(f"Loading {len(msfiles)} files from {full_path}")
        
        for f in msfiles:
            timespan = self._parse_timespan(os.path.basename(f))
            self.paths[timespan.start_datetime] = f
            for idx in self.channel_numbers:
                channel = self._parse_channel(idx)
                key = str(timespan)  # DateTimeRange is not hashable
                self.channels[key].append(channel)
                
        tlog.log(
            f"Init: {len(self.channels)} timespans and {len(set([str(i) for ch in self.channels.values() for i in ch]))} channels"
        )

    def _ensure_array_loaded(self, date_range: DateTimeRange):
        """Ensure data for the specified date range is loaded"""
        key = str(date_range)
        if key not in self.channels or date_range.start_datetime not in self.paths:
            dt = date_range.end_datetime - date_range.start_datetime
            for d in range(0, dt.seconds // 60):
                date = date_range.start_datetime + timedelta(minutes=d)
                if self.date_range is None or date in self.date_range:
                    continue
                date_path = self._get_datepath(date)
                full_path = fs_join(self.path, date_path)
                self._load_channels(full_path)

    def get_channels(self, date_range: DateTimeRange) -> List[Channel]:
        """Get list of channels available for the specified date range"""
        self._ensure_array_loaded(date_range)
        tmp_channels = self.channels.get(str(date_range), [])
        stations = set(map(lambda c: c.station, tmp_channels))
        _ = list(map(lambda s: self._validate_station(s), stations))
        logger.info(f"Getting {len(tmp_channels)} channels for {date_range}")
        return tmp_channels

    def get_timespans(self) -> List[DateTimeRange]:
        """Get list of all available timespans"""
        if self.date_range is not None:
            minutes = (
                int((self.date_range.end_datetime - self.date_range.start_datetime).total_seconds()) // 60
            )
            return [
                DateTimeRange(
                    self.date_range.start_datetime.replace(tzinfo=timezone.utc) + timedelta(minutes=d),
                    self.date_range.start_datetime.replace(tzinfo=timezone.utc) + timedelta(minutes=d + 1),
                )
                for d in range(0, minutes)
            ]
        return list([DateTimeRange.from_range_text(d) for d in sorted(self.channels.keys())])

    def read_data(self, timespan: DateTimeRange, chan: Channel) -> ChannelData:
        """Read data for a specific channel and timespan"""
        self._ensure_array_loaded(timespan)
        
        # Reconstruct the file name from the channel parameters
        filename = self._get_filename(timespan)
        number = int(chan.station.name)
        
        if not self.fs.exists(filename):
            logger.warning(f"Could not find file {filename}")
            return ChannelData.empty()

        try:
            with h5py.File(filename, "r") as f:
                # Read data for the specific channel
                data = f["/Acquisition/Raw[0]/RawData"][:, number]
                starttime = f["/Acquisition/Raw[0]/RawDataTime"][0] / 1e6
                
                # Create obspy trace
                trace = obspy.Trace(data)
                trace.stats.network = chan.station.network
                trace.stats.station = chan.station.name
                trace.stats.channel = chan.type.name
                trace.stats.sampling_rate = self.sampling_rate
                trace.stats.starttime = starttime
                
                stream = obspy.Stream([trace])
                
        except Exception as e:
            logger.error(f"Error reading data from {filename}: {e}")
            return ChannelData.empty()
            
        return ChannelData(stream)

    def _parse_channel(self, cha_number: int) -> Channel:
        """Parse channel number into Channel object"""
        cha_number = str(cha_number).zfill(5)
        return Channel(
            ChannelType("XXZ"),
            Station(self.array_name, cha_number),
        )

    def _parse_timespan(self, filename: str) -> DateTimeRange:
        """Parse filename to extract timespan"""
        starttime = datetime.strptime(filename, self.file_naming).replace(tzinfo=timezone.utc)
        return DateTimeRange(starttime, starttime + timedelta(minutes=1))  # assuming one minute file

    def _get_filename(self, timespan: DateTimeRange) -> str:
        """Get filename for a specific timespan"""
        return self.paths.get(timespan.start_datetime, "")

    def _get_datepath(self, date: datetime) -> str:
        """Get date path for a specific date"""
        return datetime.strftime(date, self.file_naming)

    def _validate_station(self, station: Station):
        """Validate station by setting required coordinates"""
        # To pass Station.valid(), otherwise no channel is returned
        station.lat = 0.0
        station.lon = 0.0
        station.elevation = 0.0

    def get_inventory(self, ts, station) -> obspy.Inventory:
        """Get inventory for station (returns empty inventory)"""
        return obspy.Inventory()
        
    def discover_files(self, pattern: str = "*.h5") -> List[str]:
        """
        Discover H5 files matching the given pattern using glob.
        
        Parameters:
            pattern: glob pattern to match files (default: "*.h5")
            
        Returns:
            List of file paths matching the pattern
        """
        full_pattern = fs_join(self.path, pattern)
        files = self.fs.glob(full_pattern)
        logger.info(f"Discovered {len(files)} files matching pattern '{pattern}'")
        return sorted(files)