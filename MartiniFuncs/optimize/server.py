"""HTTP front end for interactive optimization: serves a page and runs jobs on worker processes.

    python -m MartiniFuncs.optimize.server --page MODULE:FUNCTION \\
        --worker uma=/opt/envs/uma/bin/python:uma,aimnet,gfn2 \\
        --worker mace=/opt/envs/mace/bin/python:mace [--host 127.0.0.1] [--port 8000]

--page names a function returning the page HTML (e.g. a chemistry submodule's
builder page). Each --worker is NAME=PYTHON:MODELS: a worker process
(optimize.worker) started with that Python executable, serving those models;
without --worker, one worker runs in this Python with every model it can load.
Workers see this package through PYTHONPATH, so a conda environment without
MartiniFuncs installed works too. Only the standard library is used here.

Endpoints (JSON):
    GET  /                      the page
    GET  /models                {model: {"label", "available", "reason", "worker"}}
    POST /optimize              request (see optimize.worker) -> {"job": ID}
    GET  /jobs/ID?since=N       {"status", "message", "frames": frames N.., "count", ...}
    POST /jobs/ID/cancel        cancel a running job
One job runs per worker at a time; a second request for a busy worker is refused (409).
"""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import import_module
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from urllib.parse import parse_qs, urlparse
import uuid

import MartiniFuncs
from MartiniFuncs.optimize.engine import MODELS

FINAL = ("converged", "not_converged", "cancelled", "error")


class Job:
    def __init__(self, model, worker):
        self.id, self.model, self.worker = uuid.uuid4().hex[:10], model, worker
        self.status, self.message, self.frames, self.steps = "queued", "queued", [], None
        self.lock = threading.Lock()

    def state(self, since=0):
        with self.lock:
            return {"job": self.id, "model": self.model, "status": self.status, "message": self.message,
                    "steps": self.steps, "count": len(self.frames), "frames": self.frames[since:]}


class Worker:
    """One optimize.worker process and the jobs it runs."""

    def __init__(self, name, python, models):
        self.name, self.models, self.status, self.active = name, list(models), {}, None
        self.ready = threading.Event()
        environment = dict(os.environ)
        package_root = str(Path(MartiniFuncs.__file__).resolve().parents[1])
        environment["PYTHONPATH"] = os.pathsep.join(filter(None, [package_root, environment.get("PYTHONPATH")]))
        self.process = subprocess.Popen([python, "-m", "MartiniFuncs.optimize.worker", "--models", ",".join(models)],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1,
                                        env=environment)
        self.jobs, self.write_lock = {}, threading.Lock()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.process.stdout:
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                continue      # stray library output
            kind = message.get("type")
            if kind == "hello":
                self.status = message["models"]
                self.ready.set()
                continue
            job = self.jobs.get(message.get("job"))
            if job is None:
                continue
            with job.lock:
                if kind == "status":
                    job.status, job.message = "running", message["message"]
                elif kind == "frame":
                    job.frames.append({k: v for k, v in message.items() if k not in ("type", "job")})
                elif kind == "done":
                    job.status, job.message, job.steps = message["status"], message["message"], message.get("steps")
            if kind == "done" and self.active is job:
                self.active = None
        self.ready.set()
        for model in self.models:
            self.status.setdefault(model, {"label": MODELS.get(model, {}).get("label", model), "available": False,
                                           "reason": "worker process exited"})

    def send(self, message):
        with self.write_lock:
            self.process.stdin.write(json.dumps(message) + "\n")
            self.process.stdin.flush()

    def submit(self, request):
        if self.active is not None and self.active.status not in FINAL:
            raise RuntimeError(f"{self.name} worker is busy with job {self.active.id}; cancel it first")
        job = Job(request["model"], self.name)
        self.jobs[job.id], self.active = job, job
        self.send({"type": "run", "job": job.id, "request": request})
        return job

    def close(self):
        try:
            self.send({"type": "quit"})
        except (BrokenPipeError, OSError, ValueError):
            pass
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()


class Service:
    def __init__(self, page, workers):
        self.page, self.workers = page, workers

    def models(self):
        found = {}
        for worker in self.workers:
            worker.ready.wait(timeout=120)
            for name, status in worker.status.items():
                if name not in found or (status["available"] and not found[name]["available"]):
                    found[name] = {**status, "worker": worker.name}
        return found

    def worker_for(self, model):
        status = self.models().get(model)
        if status is None or not status["available"]:
            raise LookupError(f"model {model!r} is not available"
                              + (f": {status['reason']}" if status else ""))
        return next(w for w in self.workers if w.name == status["worker"])

    def job(self, job_id):
        for worker in self.workers:
            if job_id in worker.jobs:
                return worker, worker.jobs[job_id]
        raise KeyError(job_id)


def make_handler(service):
    class Handler(BaseHTTPRequestHandler):
        def _reply(self, code, body, content_type="application/json"):
            data = body.encode("utf-8") if isinstance(body, str) else json.dumps(body).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", content_type + "; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, fmt, *args):      # quiet: only errors
            if args and str(args[1]).startswith(("4", "5")):
                super().log_message(fmt, *args)

        def do_GET(self):
            url = urlparse(self.path)
            parts = [p for p in url.path.split("/") if p]
            if not parts:
                return self._reply(200, service.page, "text/html")
            if parts == ["models"]:
                return self._reply(200, service.models())
            if len(parts) == 2 and parts[0] == "jobs":
                try:
                    _, job = service.job(parts[1])
                except KeyError:
                    return self._reply(404, {"error": "no such job"})
                since = int(parse_qs(url.query).get("since", ["0"])[0])
                return self._reply(200, job.state(since))
            return self._reply(404, {"error": "not found"})

        def do_POST(self):
            parts = [p for p in urlparse(self.path).path.split("/") if p]
            length = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                return self._reply(400, {"error": "body must be JSON"})
            if parts == ["optimize"]:
                try:
                    job = service.worker_for(body.get("model")).submit(body)
                except LookupError as exc:
                    return self._reply(400, {"error": str(exc)})
                except RuntimeError as exc:
                    return self._reply(409, {"error": str(exc)})
                return self._reply(200, {"job": job.id})
            if len(parts) == 3 and parts[0] == "jobs" and parts[2] == "cancel":
                try:
                    worker, job = service.job(parts[1])
                except KeyError:
                    return self._reply(404, {"error": "no such job"})
                worker.send({"type": "cancel", "job": job.id})
                return self._reply(200, {"job": job.id, "cancel": "requested"})
            return self._reply(404, {"error": "not found"})

    return Handler


def parse_worker(spec):
    """NAME=PYTHON:MODELS -> (name, python, [models])."""
    name, _, rest = spec.partition("=")
    python, _, models = rest.rpartition(":")
    if not name or not python or not models:
        raise argparse.ArgumentTypeError(f"--worker must be NAME=PYTHON:MODEL[,MODEL], got {spec!r}")
    return name, python, [m.strip() for m in models.split(",") if m.strip()]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--page", required=True, help="MODULE:FUNCTION returning the page HTML")
    parser.add_argument("--worker", action="append", type=parse_worker, default=[],
                        help="NAME=PYTHON:MODEL[,MODEL] (repeatable)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    module, _, function = args.page.partition(":")
    print("Rendering the page...", flush=True)
    page = getattr(import_module(module), function)()
    specs = args.worker or [("local", sys.executable, list(MODELS))]
    workers = [Worker(*spec) for spec in specs]
    service = Service(page, workers)
    for name, status in service.models().items():
        print(f"  {name:7s} {'available' if status['available'] else 'unavailable: ' + status['reason']}"
              f" ({status['worker']} worker)", flush=True)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(service))
    print(f"Serving on http://{args.host}:{args.port}/  (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        for worker in workers:
            worker.close()
        time.sleep(0.1)


if __name__ == "__main__":
    main()
