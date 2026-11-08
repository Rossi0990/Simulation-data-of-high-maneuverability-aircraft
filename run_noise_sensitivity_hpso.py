"""Run HPSO noise sensitivity with the shared 0-29 seed convention.

The actual HPSO application entry point is ``aircraft_kf_experiment.py``;
``HPSO.py`` is the lower-level DEAP implementation and does not expose the
noise-sensitivity CLI.  Reuse the established HPSO sensitivity runner so the
training/test noise seed mapping remains consistent.
"""

import sys

from run_noise_sensitivity import main


if __name__ == "__main__":
    if "--output" not in sys.argv:
        sys.argv.extend(["--output", "results/aircraft_kf_sensitivity/hpso"])
    main()
