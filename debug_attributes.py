
import torch
import sys
import os

# Mock the callback to avoid import error
from benchmarl.experiment.callback import Callback
class WinRateReportDebounced(Callback):
    def __init__(self, *args, **kwargs): pass
    def on_setup(self): pass
    def on_batch_collected(self, batch): pass

# Add BenchMARL to path if needed (it is in current dir)
sys.path.append(os.getcwd())

from benchmarl.experiment import Experiment

checkpoint_path = "BenchMARL/outputs/2026-02-04_16-08-46/ensemblealgorithm_layup_ensemblemodel__7f8f0570_26_02_04-16_08_46/checkpoints/checkpoint_9750000.pt"
print(f"Loading {checkpoint_path}...")
exp = Experiment.reload_from_file(checkpoint_path)
loss = exp.losses["attacker"]
print("Loss type:", type(loss))
print("Attributes:", dir(loss))
