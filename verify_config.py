import pandas as pd
import numpy as np
from pmecg.utils.data import _apply_configuration, SUPPORTED_LEADS

df = pd.DataFrame(np.zeros((1000, 12)), columns=SUPPORTED_LEADS)
config = ['I', 'II', 'V2']
result = _apply_configuration(df, config)
print(f"Number of rows for config {config}: {len(result)}")
for i, (sig, leads) in enumerate(result):
    print(f"Row {i}: leads={leads}, signal shape={sig.shape}")

config_mixed = [['I', 'V1'], 'II']
result_mixed = _apply_configuration(df, config_mixed)
print(f"\nNumber of rows for config {config_mixed}: {len(result_mixed)}")
for i, (sig, leads) in enumerate(result_mixed):
    print(f"Row {i}: leads={leads}, signal shape={sig.shape}")
