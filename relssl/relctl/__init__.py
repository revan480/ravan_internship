"""relctl — interactive control panel for the relssl SSL repo.

Run it from the repo root inside the project's conda env:

    python -m relssl.relctl              # auto: Rich tier if installed, else plain
    python -m relssl.relctl --plain      # force the zero-dependency plain tier
    python -m relssl.relctl --validate   # check the knob catalog against the configs
"""

__all__ = ["app", "config", "actions", "jobs", "knobs", "preflight", "ui"]
