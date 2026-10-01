"""Stage logging with an elapsed-time heartbeat, not an invented percentage."""
from datetime import datetime
from threading import Event, Thread
from time import perf_counter


def log(message):
    """Print a timestamped, flushed line suitable for a redirected Slurm log."""
    print(f'[{datetime.now().isoformat(timespec="seconds")}] {message}', flush=True)


def run_stage(label, function, *args, progress=log, heartbeat_seconds=30, **kwargs):
    """Call a stage, logging start, elapsed heartbeat, completion or failure.

    Heartbeats indicate the call has not returned; they are not evidence of CPU
    activity or a completion estimate. The thread never accesses scientific data.
    progress=None disables logs and avoids creating a background thread.
    """
    if progress is None:
        return function(*args, **kwargs)
    if heartbeat_seconds <= 0:
        raise ValueError('heartbeat_seconds must be positive')
    started = perf_counter()
    stop = Event()

    def heartbeat():
        while not stop.wait(heartbeat_seconds):
            progress(f'{label}: still running; elapsed {perf_counter()-started:.1f} s')

    progress(f'{label}: starting')
    thread = Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        result = function(*args, **kwargs)
    except BaseException:
        stop.set()
        thread.join()
        progress(f'{label}: failed after {perf_counter()-started:.2f} s')
        raise
    else:
        stop.set()
        thread.join()
        progress(f'{label}: completed in {perf_counter()-started:.2f} s')
        return result
