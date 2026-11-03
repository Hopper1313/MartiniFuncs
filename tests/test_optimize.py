"""Interactive optimization: engine constraints, the worker protocol and the HTTP server (EMT, no ML models)."""
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import unittest
import urllib.error
import urllib.request

import numpy as np
from ase.calculators.emt import EMT

from MartiniFuncs.chemistry.anthracene_ma.build import PairFrames, build_pair
from MartiniFuncs.optimize.engine import Cancelled, ModelCache, RestrainedCalculator, model_status, optimize

ROOT = str(Path(__file__).resolve().parents[1])


def stacked_pair():
    frames = PairFrames()
    atoms, _ = build_pair(distance=3.0, frames=frames)
    n = len(frames.anthracene_xyz)
    pairs = [[int(frames.termini[0]), n + int(frames.alkene[0])], [int(frames.termini[1]), n + int(frames.alkene[1])]]
    return atoms, pairs


class EngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.atoms, cls.pairs = stacked_pair()

    def run_mode(self, constraint, optimizer="BFGS", steps=30):
        frames = []
        final, converged, n = optimize(self.atoms.get_chemical_symbols(), self.atoms.positions, calculator=EMT(),
                                       constraint=constraint, optimizer=optimizer, steps=steps, callback=frames.append)
        return final, frames, n

    def distances(self, atoms):
        return [atoms.get_distance(*p) for p in self.pairs]

    def test_constraint_modes(self):
        start = self.distances(self.atoms)
        free, frames, n = self.run_mode({"mode": "none"})
        self.assertEqual([f["step"] for f in frames], list(range(n + 1)))       # start geometry once, then every step
        self.assertLess(max(self.distances(free)), min(start) - 0.5)            # EMT pulls the free pair together
        fixed, _, _ = self.run_mode({"mode": "fix_distances", "pairs": self.pairs})
        assert_close = np.testing.assert_allclose
        assert_close(self.distances(fixed), start, atol=1e-6)
        restrained, frames, _ = self.run_mode({"mode": "restrain_distances", "pairs": self.pairs, "k": 50.0})
        assert_close(self.distances(restrained), start, atol=0.05)
        self.assertGreater(frames[-1]["restraint_energy"], 0)
        held = [i for pair in self.pairs for i in pair]
        pinned, _, _ = self.run_mode({"mode": "fix_atoms", "atoms": held})
        assert_close(pinned.positions[held], self.atoms.positions[held], atol=1e-12)
        sella, _, _ = self.run_mode({"mode": "fix_distances", "pairs": self.pairs}, optimizer="sella_min", steps=10)
        assert_close(self.distances(sella), start, atol=1e-4)

    def test_restraint_energy_and_forces(self):
        """RestrainedCalculator adds 0.5 k (d - d0)^2 and its exact gradient."""
        atoms = self.atoms.copy()
        pair, k = self.pairs[0], 7.0
        target = atoms.get_distance(*pair) - 0.3
        atoms.calc = RestrainedCalculator(EMT(), [pair], [target], k)
        energy, forces = atoms.get_potential_energy(), atoms.get_forces()
        plain = self.atoms.copy()
        plain.calc = EMT()
        self.assertAlmostEqual(energy - plain.get_potential_energy(), 0.5 * k * 0.3 ** 2, places=9)
        h, i = 1e-5, pair[1]
        for axis in range(3):
            shifted = [atoms.copy() for _ in range(2)]
            for sign, copy in zip((1, -1), shifted):
                copy.positions[i, axis] += sign * h
                copy.calc = RestrainedCalculator(EMT(), [pair], [target], k)
            numeric = -(shifted[0].get_potential_energy() - shifted[1].get_potential_energy()) / (2 * h)
            self.assertAlmostEqual(forces[i, axis], numeric, places=5)

    def test_cancel_and_validation(self):
        def stop(frame):
            if frame["step"] == 2:
                raise Cancelled()
        with self.assertRaises(Cancelled):
            optimize(["C", "O"], [[0, 0, 0], [1.3, 0, 0]], calculator=EMT(), optimizer="FIRE", callback=stop)
        for constraint in ({"mode": "fix_atoms"}, {"mode": "fix_distances", "pairs": [[0, 0]]},
                           {"mode": "fix_atoms", "atoms": [5]}, {"mode": "bogus"}):
            with self.assertRaises(ValueError):
                optimize(["C", "O"], [[0, 0, 0], [1.3, 0, 0]], calculator=EMT(), constraint=constraint)
        with self.assertRaises(ValueError):
            optimize(["C", "O"], [[0, 0, 0], [1.3, 0, 0]], calculator=EMT(), optimizer="nope")

    def test_model_status(self):
        status = model_status(["emt", "uma"])
        self.assertTrue(status["emt"]["available"])
        os.environ.pop("UMA_CHECKPOINT", None)
        self.assertFalse(model_status(["uma"])["uma"]["available"])
        self.assertIsNone(ModelCache()._load("emt"))


class WorkerTests(unittest.TestCase):
    def test_protocol(self):
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [ROOT, os.environ.get("PYTHONPATH")])))
        process = subprocess.Popen([sys.executable, "-m", "MartiniFuncs.optimize.worker", "--models", "emt"],
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                   text=True, bufsize=1, env=env)
        self.addCleanup(process.kill)
        hello = json.loads(process.stdout.readline())
        self.assertEqual(hello["type"], "hello")
        self.assertTrue(hello["models"]["emt"]["available"])
        request = {"model": "emt", "symbols": ["C", "O"], "positions": [[0, 0, 0], [1.3, 0, 0]], "steps": 20}
        process.stdin.write(json.dumps({"type": "run", "job": "a", "request": request}) + "\n")
        process.stdin.flush()     # stdin stays open while the job runs (the Windows stall this guards against)
        messages = []
        while not messages or messages[-1]["type"] != "done":
            messages.append(json.loads(process.stdout.readline()))
        kinds = [m["type"] for m in messages]
        self.assertIn("frame", kinds)
        self.assertEqual(messages[-1]["status"], "converged")
        process.stdin.write(json.dumps({"type": "quit"}) + "\n")
        process.stdin.flush()
        self.assertEqual(process.wait(timeout=30), 0)


class ServerTests(unittest.TestCase):
    def test_page_models_and_a_job(self):
        from http.server import ThreadingHTTPServer
        from MartiniFuncs.optimize.server import Service, Worker, make_handler, parse_worker
        self.assertEqual(parse_worker(r"uma=C:\envs\uma\python.exe:uma,emt"), ("uma", r"C:\envs\uma\python.exe", ["uma", "emt"]))
        worker = Worker("local", sys.executable, ["emt"])
        self.addCleanup(worker.close)
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(Service("<p>page</p>", [worker])))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.shutdown)
        base = f"http://127.0.0.1:{server.server_address[1]}"

        def call(path, body=None):
            data = None if body is None else json.dumps(body).encode()
            request = urllib.request.Request(base + path, data=data, method="GET" if body is None else "POST",
                                             headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = response.read().decode()
            return payload if path == "/" else json.loads(payload)

        self.assertEqual(call("/"), "<p>page</p>")
        self.assertTrue(call("/models")["emt"]["available"])
        atoms, pairs = stacked_pair()
        job = call("/optimize", {"model": "emt", "symbols": atoms.get_chemical_symbols(),
                                 "positions": atoms.positions.tolist(), "steps": 15,
                                 "constraint": {"mode": "fix_distances", "pairs": pairs}})["job"]
        for _ in range(300):
            state = call(f"/jobs/{job}")
            if state["status"] in ("converged", "not_converged", "cancelled", "error"):
                break
            time.sleep(0.1)
        self.assertIn(state["status"], ("converged", "not_converged"), state["message"])
        self.assertEqual(state["count"], len(state["frames"]))
        final = np.array(state["frames"][-1]["positions"])
        np.testing.assert_allclose([np.linalg.norm(final[i] - final[j]) for i, j in pairs],
                                   [atoms.get_distance(*p) for p in pairs], atol=1e-6)
        self.assertEqual(call(f"/jobs/{job}?since={state['count']}")["frames"], [])
        with self.assertRaises(urllib.error.HTTPError) as unknown:
            call("/optimize", {"model": "uma", "symbols": ["C"], "positions": [[0, 0, 0]]})
        self.assertEqual(unknown.exception.code, 400)

    def test_results_store(self):
        """Saved results: POST/GET/DELETE over HTTP, one JSON file each, reloaded by a new store."""
        import tempfile
        from http.server import ThreadingHTTPServer
        from MartiniFuncs.optimize.server import ResultStore, Service, make_handler
        folder = Path(tempfile.mkdtemp())
        self.addCleanup(__import__("shutil").rmtree, folder, True)
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(Service("<p>page</p>", [], ResultStore(folder))))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.shutdown)
        base = f"http://127.0.0.1:{server.server_address[1]}"

        def call(path, body=None, method=None):
            data = None if body is None else json.dumps(body).encode()
            request = urllib.request.Request(base + path, data=data, method=method or ("GET" if body is None else "POST"),
                                             headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.loads(response.read().decode())

        state = call("/results")
        self.assertEqual((state["persistent"], state["results"]), (True, []))
        first = {"id": "r1", "name": "first", "created": "2026-11-03T10:00:00Z", "energy": -1.5, "positions": [[0, 0, 0]]}
        second = {"id": "r0", "name": "second", "created": "2026-11-03T11:00:00Z"}
        call("/results", first)
        call("/results", second)
        self.assertEqual([r["id"] for r in call("/results")["results"]], ["r1", "r0"])     # oldest first
        call("/results", {**first, "name": "renamed"})
        self.assertEqual(call("/results/r1")["name"], "renamed")
        self.assertEqual(sorted(p.name for p in folder.glob("*.json")), ["r0.json", "r1.json"])
        self.assertEqual(ResultStore(folder).get("r1")["name"], "renamed")
        self.assertEqual(call("/results/r0", method="DELETE"), {"deleted": "r0"})
        self.assertEqual([p.name for p in folder.glob("*.json")], ["r1.json"])
        for path, body, method in (("/results", {"id": "../x"}, None), ("/results/r0", None, "DELETE"),
                                   ("/results/nope", None, None)):
            with self.assertRaises(urllib.error.HTTPError) as error:
                call(path, body, method)
            self.assertEqual(error.exception.code, 400 if body else 404)
        # Static assets (--assets): served from the folder only.
        assets = folder / "assets"
        (assets / "x3dom").mkdir(parents=True)
        (assets / "x3dom" / "x3dom.js").write_text("var x3dom = {};", encoding="utf-8")
        service = Service("<p>page</p>", [], ResultStore(), assets)
        self.assertEqual(service.asset(["x3dom", "x3dom.js"])[0], b"var x3dom = {};")
        self.assertIn("javascript", service.asset(["x3dom", "x3dom.js"])[1])
        for parts in (["..", "r1.json"], ["x3dom", "missing.js"], ["x3dom"]):
            self.assertIsNone(service.asset(parts), parts)
        self.assertIsNone(Service("", []).asset(["x3dom", "x3dom.js"]))
        with self.assertRaises(urllib.error.HTTPError) as error:
            call("/assets/x3dom/x3dom.js")          # this server has no --assets
        self.assertEqual(error.exception.code, 404)
        memory = ResultStore()
        memory.save(first)
        self.assertEqual(memory.state()["persistent"], False)


if __name__ == "__main__":
    unittest.main()
