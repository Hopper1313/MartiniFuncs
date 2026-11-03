"""Interactive constrained geometry optimization: engine (models, constraints, optimizers),
worker (one process per Python environment, JSON lines over stdin/stdout) and server
(HTTP front end that serves a builder page and dispatches jobs to the workers).
Nothing here names a reaction system; the page comes from a chemistry submodule.
"""
