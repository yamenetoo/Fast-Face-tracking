"""Run the complete simulation and print the main tables.

    python examples/run_simulation.py [output_folder]
"""
import sys

import fastfacetrack as fft

out = sys.argv[1] if len(sys.argv) > 1 else "results"
result = fft.run_experiment(fft.ExperimentConfig(out_dir=out))
print(result["aggregate"].reset_index()[["detector", "model", "det_f1_mean", "detect_ms_mean", "classify_ms_mean"]]
      .round(3).to_string(index=False))
