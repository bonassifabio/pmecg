import pandas as pd
import pandas.errors
from ptbxl_helper import get_ptbxl_data

from pmecg.plot import ECGInformation, ECGPlotter

# ── Load ECG ──────────────────────────────────────────────────────────────
record, metadata, stats = get_ptbxl_data(1)
df = pd.DataFrame(record.p_signal, columns=record.sig_name)
sampling_freq = record.fs

ecg_info = ECGInformation(
    # patient_name="John Doe",
    age=metadata["age"],
    sex=metadata["sex"],
    date=metadata["date"],
    machine_model="PTB-XL ECG Machine",
    filter="Some filter",
)

custom_configuration = record.sig_name
ECGPlotter(grid_mode="cm", print_information=True).plot(
    df, custom_configuration, sampling_frequency=sampling_freq, show=True, information=ecg_info, stats=stats
)
