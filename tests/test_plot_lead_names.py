import pandas as pd
import pandas.errors
from ptbxl_helper import get_ptbxl_data

from pmecg import LeadsMap
from pmecg.plot import ECGInformation, ECGPlotter

# ── Load ECG ──────────────────────────────────────────────────────────────
record, metadata, stats = get_ptbxl_data(1)
lead_names = record.sig_name
lead_names[3] = "-aVR"
df = pd.DataFrame(record.p_signal, columns=lead_names)
sampling_freq = record.fs

ecg_info = ECGInformation(
    # patient_name="John Doe",
    age=metadata["age"],
    sex=metadata["sex"],
    date=metadata["date"],
    machine_model="PTB-XL ECG Machine",
    filter="Some filter",
)

custom_configuration = lead_names
leads_map = LeadsMap(AVR="-aVR")
ECGPlotter(grid_mode="cm", print_information=True).plot(
    df, "4x3", sampling_frequency=sampling_freq, show=True, information=ecg_info, stats=stats, leads_map=leads_map
)
