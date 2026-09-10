"""
Regenerates data/ucs/scaler_params.yaml from ucs_windows.parquet (post-purge train split).
"""
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from collections import OrderedDict

ws = Path(r"c:\Users\Vicky\Documents\SIH_2026\LSTM-type-model-CICIDS2018-UCS")

UCS_WINDOWS = ws / "data/ucs/ucs_windows.parquet"
OUTPUT_SCALER_PARAMS = ws / "data/ucs/scaler_params.yaml"

UCS_METADATA_COLUMNS = {
    "window_id", "window_start_utc", "window_end_utc", "source_day", "split"
}
UCS_LABEL_COLUMNS = {
    "label_binary", "label_attack_type", "future_attack_label", "has_malicious_flows", "raw_label_dominant"
}
MASK_COLUMNS = {
    "mask_has_traffic_volume_features", "mask_has_flow_timing_features",
    "mask_has_packet_level_features", "mask_has_tcp_flags",
    "mask_has_graph_topology", "mask_has_identity_auth"
}

LOG1P_BASE_COLUMNS = {
    "byte_count_fwd", "byte_count_bwd", "packet_count_fwd", "packet_count_bwd",
    "bytes_per_sec", "packets_per_sec", "duration_sec"
}

def main():
    df = pd.read_parquet(UCS_WINDOWS)
    train_df = df[df["split"] == "train"].copy()
    print(f"Post-purge train split shape: {train_df.shape}")

    excluded = UCS_METADATA_COLUMNS | UCS_LABEL_COLUMNS | MASK_COLUMNS
    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in excluded]
    print(f"Total numeric feature columns to scale: {len(feature_cols)}")
    assert len(feature_cols) == 400, f"Expected 400 feature columns, got {len(feature_cols)}"

    scaler_dict = OrderedDict()

    for col in feature_cols:
        is_log1p = any(col.startswith(base) for base in LOG1P_BASE_COLUMNS)
        vals = train_df[col].to_numpy(dtype=float)
        if is_log1p:
            vals = np.log1p(np.maximum(0.0, vals))

        q25 = float(np.percentile(vals, 25))
        median = float(np.median(vals))
        q75 = float(np.percentile(vals, 75))
        scale = q75 - q25
        if scale == 0.0:
            scale = 1.0

        scaler_dict[col] = {
            "median": median,
            "scale": scale,
            "q25": q25,
            "q75": q75,
            "is_log1p": is_log1p
        }

    # Representer for OrderedDict
    def represent_ordereddict(dumper, data):
        return dumper.represent_mapping("tag:yaml.org,2002:map", data.items())

    yaml.add_representer(OrderedDict, represent_ordereddict)

    with open(OUTPUT_SCALER_PARAMS, "w", encoding="utf-8") as f:
        yaml.dump(dict(scaler_dict), f, default_flow_style=False, sort_keys=False)

    print(f"[+] Successfully regenerated scaler parameters to {OUTPUT_SCALER_PARAMS}")

    # Print check for the 12 packet features
    packet_12 = [
        "pkt_ttl_min", "pkt_ttl_max", "pkt_ttl_std", "pkt_ttl_mode",
        "pkt_frag_mf_count", "pkt_frag_df_count",
        "pkt_payload_size_p25", "pkt_payload_size_p50",
        "pkt_payload_size_p75", "pkt_payload_size_p95",
        "pkt_tcp_retrans_count", "pkt_port_scan_seq_score"
    ]
    print("\nUpdated scaler params for 12 packet features:")
    for p in packet_12:
        print(f"  {p}: median={scaler_dict[p]['median']}, scale={scaler_dict[p]['scale']}")

if __name__ == "__main__":
    main()
