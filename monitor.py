import logging
import signal
import os
from mpi4py import MPI


class Monitor:

    def __init__(self):
        self.canceled = False
        self.mpi = MPI.COMM_WORLD
        self.rank = self.mpi.Get_rank()
        self._register_signal_handlers()

    def _register_signal_handlers(self):
        def cancel(signum, frame):
            logging.warning(f'Process {os.getpid()} with rank {self.rank} received cancelation signal, aborting.')
            self.canceled = True

        signal.signal(signal.SIGUSR1, cancel)
        signal.signal(signal.SIGTERM, cancel)
        signal.signal(signal.SIGINT, cancel)
        signal.signal(signal.SIGHUP, cancel)

    @property
    def should_quit(self):
        self.canceled = self.mpi.allreduce(self.canceled, MPI.LOR)
        if self.canceled:
            logging.warning(f'Process {os.getpid()} with rank {self.rank} is now aware of cancelation request.')
        return self.canceled
