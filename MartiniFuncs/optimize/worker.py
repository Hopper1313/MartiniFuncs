"""One optimization worker: serves the models of one Python environment over stdin/stdout.

    python -m MartiniFuncs.optimize.worker --models uma,aimnet,gfn2

Protocol: one JSON object per line. The worker first prints
{"type": "hello", "models": {name: {"label", "available", "reason"}}}, then reads

    {"type": "run", "job": ID, "request": {"model", "symbols", "positions", "charge",
                                           "spin", "constraint", "optimizer", "fmax", "steps"}}
    {"type": "cancel", "job": ID}
    {"type": "quit"}

and prints {"type": "status", "job", "message"}, one {"type": "frame", "job", ...} per
optimization step (engine.optimize's frames) and finally
{"type": "done", "job", "status": "converged" | "not_converged" | "cancelled" | "error",
 "message", "steps"}. Models load on first use and stay loaded. One job runs at a time.
"""
import argparse
import json
import os
import sys
import threading
import traceback

from MartiniFuncs.optimize.engine import Cancelled, ModelCache, model_status, optimize

_print_lock = threading.Lock()
_channel = {}


def _private_channel():
    """Protocol streams on duplicated handles; the process's own stdin/stdout become devnull/stderr.

    Libraries may print (which would corrupt the JSON lines) or query stdin when
    imported; on Windows that query blocks while the main thread waits on a
    pipe read, stalling the job thread. With fds 0 and 1 replaced, neither touches
    the protocol pipes.
    """
    commands = os.fdopen(os.dup(0), "r", encoding="utf-8")
    replies = os.fdopen(os.dup(1), "w", encoding="utf-8", buffering=1)
    sys.stdout.flush()
    null = os.open(os.devnull, os.O_RDONLY)
    os.dup2(null, 0)
    os.dup2(2, 1)
    sys.stdin = open(os.devnull, encoding="utf-8")
    return commands, replies


def send(message):
    with _print_lock:
        _channel["replies"].write(json.dumps(message) + "\n")
        _channel["replies"].flush()


def run_job(job, request, cache, cancel):
    try:
        model = request["model"]
        if not cache.is_loaded(model):
            send({"type": "status", "job": job, "message": f"loading {model} (first use; can take ~30 s)"})
        calculator = cache.calculator(model, charge=request.get("charge", 0), spin=request.get("spin", 1))
        send({"type": "status", "job": job, "message": f"optimizing with {model}"})

        def frame(data):
            if cancel.is_set():
                raise Cancelled()
            send({"type": "frame", "job": job, **data})

        _, converged, steps = optimize(request["symbols"], request["positions"], calculator=calculator,
                                       constraint=request.get("constraint"), optimizer=request.get("optimizer", "BFGS"),
                                       fmax=request.get("fmax", 0.05), steps=request.get("steps", 500),
                                       charge=request.get("charge", 0), spin=request.get("spin", 1), callback=frame)
        send({"type": "done", "job": job, "status": "converged" if converged else "not_converged", "steps": steps,
              "message": f"{'converged' if converged else 'stopped at the step limit'} after {steps} steps"})
    except Cancelled:
        send({"type": "done", "job": job, "status": "cancelled", "message": "cancelled"})
    except Exception as exc:        # report any failure to the page instead of dying
        traceback.print_exc(file=sys.stderr)
        send({"type": "done", "job": job, "status": "error", "message": f"{type(exc).__name__}: {exc}"})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", required=True, help="comma-separated models this worker serves")
    args = parser.parse_args(argv)
    names = [name.strip() for name in args.models.split(",") if name.strip()]
    commands, _channel["replies"] = _private_channel()
    send({"type": "hello", "models": model_status(names)})
    cache, current, cancels = ModelCache(), None, {}
    for line in commands:
        if not line.strip():
            continue
        message = json.loads(line)
        kind = message.get("type")
        if kind == "quit":
            break
        if kind == "cancel" and message.get("job") in cancels:
            cancels[message["job"]].set()
        elif kind == "run":
            if current is not None and current.is_alive():
                send({"type": "done", "job": message["job"], "status": "error", "message": "worker busy"})
                continue
            if message["request"].get("model") not in names:
                send({"type": "done", "job": message["job"], "status": "error",
                      "message": f"this worker serves {names}"})
                continue
            cancels = {message["job"]: threading.Event()}
            current = threading.Thread(target=run_job, args=(message["job"], message["request"], cache,
                                                             cancels[message["job"]]), daemon=True)
            current.start()
    for event in cancels.values():
        event.set()


if __name__ == "__main__":
    main()
