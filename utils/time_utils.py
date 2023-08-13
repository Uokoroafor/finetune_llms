import time
from typing import Optional, Tuple


# A class for logging and printing epoch times


class EpochTimer:
    def __init__(self):
        """
        Initialize the timer object
        """
        self.start_time = None
        self.times = []

    def start(self):
        """
        Log the start time of a process
        """
        self.start_time = time.time()
        self.times.append(self.start_time)

    def lap(self):
        """
        Log the time elapsed since the start time or last lap
        """
        if self.start_time is None:
            raise Exception("EpochTimer not started")
        else:
            self.times.append(time.time())

    def get_lap_time(self, lap: int) -> Tuple[int, int, int]:
        """
        Get the time elapsed between the start time and the lap time
        """
        if lap == 0:
            return 0, 0, 0
        else:
            return self._calculate_lap_time(self.times[lap - 1], self.times[lap])

    def get_total_time(self) -> Tuple[int, int, int]:
        """
        Get the total time elapsed between the start time and the last lap
        """
        if self.start_time is None:
            raise Exception("EpochTimer not started")
        else:
            return self._calculate_lap_time(self.start_time, self.times[-1])

    def get_last_epoch_time(self) -> Tuple[int, int, int]:
        """
        Get the total time elapsed between the start time and the last lap
        """
        if self.start_time is None:
            raise Exception("EpochTimer not started")
        else:
            return self._calculate_lap_time(self.times[-2], self.times[-1])

    def print_last_epoch_time(self, label: Optional[str] = None):
        """
        Print the time elapsed between the start time and the last lap
        """
        return self.print_time(self.get_last_epoch_time(), label)

    def print_total_time(self, label: Optional[str] = None):
        """
        Print the total time elapsed between the start time and the last lap
        """
        return self.print_time(self.get_total_time(), label)

    @staticmethod
    def print_time(
        time_tuple: Tuple[int, int, int], label: Optional[str] = None
    ) -> str:
        """
        Print the time elapsed between two times
        """
        if label is None:
            label = "Time elapsed: "

        return f"{label} {time_tuple[0]:02d} hour(s) {time_tuple[1]:02d} minute(s) {time_tuple[2]:02d} second(s)"

    @staticmethod
    def _calculate_lap_time(time1: float, time2: float) -> Tuple[int, int, int]:
        """
        Calculate the time elapsed between two times
        """
        total_time = int(time2 - time1)

        hours = total_time // 3600
        minutes = (total_time % 3600) // 60
        seconds = total_time % 60

        return hours, minutes, seconds
