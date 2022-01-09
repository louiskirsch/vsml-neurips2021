import subprocess
import signal
import functools
import sys
import os


def handle_signal(proc, sig, _):
    print(f'Sending SIGINT to job {proc.pid}')
    os.kill(proc.pid, signal.SIGINT)


p = subprocess.Popen(sys.argv[1:], stdin=0, stdout=1, stderr=2)
handler = functools.partial(handle_signal, p)
signal.signal(signal.SIGTERM, handler)
signal.signal(signal.SIGUSR1, handler)
signal.signal(signal.SIGINT, handler)

ret = p.wait()

if ret < 0:
    sys.exit(1)
