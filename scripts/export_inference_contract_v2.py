"""
Export the exact inference normalization contract v2 for the probabilistic LSTM v2 checkpoint.
Produces:
  artifacts/lstm/inference_scaler_v2.yaml
  artifacts/lstm/inference_feature_order_v2.json
"""
import json
import sys
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

REPRODUCIBILITY_METADATA = (
    REPO_ROOT
    / "artifacts"
    / "lstm"
    / "probabilistic_world_model_v2"
    / "reproducibility_metadata.json"
)
SCALER_PARAMS = REPO_ROOT / "data" / "ucs" / "scaler_params.yaml"
UCS_WINDOWS = REPO_ROOT / "data" / "ucs" / "ucs_windows.parquet"
CHECKPOINT = (
    REPO_ROOT
    / "artifacts"
    / "lstm"
    / "gaussian_next_state_best_v2.pt"
)

OUTPUT_SCALER = REPO_ROOT / "artifacts" / "lstm" / "inference_scaler_v2.yaml"
OUTPUT_FEATURES = REPO_ROOT / "artifacts" / "lstm" / "inference_feature_order_v2.json"

UCS_METADATA_COLUMNS = {
    "window_id", "window_start_utc", "window_end_utc", "source_day", "split"
}
UCS_LABEL_COLUMNS = {
    "label_binary", "label_attack_type", "future_attack_label", "has_malicious_flows", "raw_label_dominant"
}

LOG1P_BASE_COLUMNS = {
    "byte_count_fwd", "byte_count_bwd", "packet_count_fwd", "packet_count_bwd",
    "bytes_per_sec", "packets_per_sec", "duration_sec"
}

def fail(msg: str) -> None:
    print(f"\n[FAIL] {msg}", file=sys.stderr)
    sys.exit(1)

def check(condition: bool, description: str) -> None:
    if condition:
        print(f"  [OK] {description}")
    else:
        fail(f"FAILED CHECK: {description}")

def main():
    print("=" * 72)
    print("Exporting v2 Inference Contract (LSTM World Model v2)")
    print("=" * 72)

    check(REPRODUCIBILITY_METADATA.exists(), "reproducibility_metadata.json exists")
    check(SCALER_PARAMS.exists(), "scaler_params.yaml exists")
    check(UCS_WINDOWS.exists(), "ucs_windows.parquet exists")
    check(CHECKPOINT.exists(), "gaussian_next_state_best_v2.pt exists")

    with open(REPRODUCIBILITY_METADATA, encoding="utf-8") as f:
        meta = json.load(f)

    with open(SCALER_PARAMS, encoding="utf-8") as f:
        scaler_raw = yaml.safe_load(f)

    parquet_sample = pd.read_parquet(UCS_WINDOWS).head(1)
    excluded = UCS_METADATA_COLUMNS | UCS_LABEL_COLUMNS
    parquet_numeric_features = [
        col for col in parquet_sample.select_dtypes(include=[np.number]).columns if col not in excluded
    ]

    meta_features = meta["features"]
    scaler_features = list(scaler_raw.keys())

    check(meta["input_dimension"] == 406, f"input_dimension = {meta['input_dimension']} (expected 406)")
    check(len(meta_features) == 406, f"metadata features count = {len(meta_features)} (expected 406)")
    check(len(scaler_features) == 400, f"scaler_params features count = {len(scaler_features)} (expected 400)")

    meta_set = set(meta_features)
    scaler_set = set(scaler_features)
    extra_in_meta = sorted(meta_set - scaler_set)

    expected_masks = sorted([
        "mask_has_traffic_volume_features", "mask_has_flow_timing_features",
        "mask_has_packet_level_features", "mask_has_tcp_flags",
        "mask_has_graph_topology", "mask_has_identity_auth"
    ])
    check(extra_in_meta == expected_masks, f"Extra features are exactly the 6 mask columns")

    meta_scaled_ordered = [f for f in meta_features if f in scaler_set]
    check(meta_scaled_ordered == scaler_features, "400 scaled features order matches")

    check(len(parquet_numeric_features) == 406, f"parquet feature count = 406")

    # Build inference_scaler_v2.yaml
    print("\nWriting inference_scaler_v2.yaml...")
    scaler_doc = OrderedDict()
    scaler_doc["_metadata"] = OrderedDict([
        ("version", "v2"),
        ("artifact_type", "inference_scaler"),
        ("description", (
            "Exact normalization parameters for live inference against the v2 "
            "probabilistic LSTM world-model checkpoint (gaussian_next_state_best_v2.pt). "
            "Includes Option A data fix (zero-filled packet telemetry & absent mask)."
        )),
        ("checkpoint", "gaussian_next_state_best_v2.pt"),
        ("checkpoint_commit", meta.get("repository_commit", "unknown")),
        ("exported_at", datetime.now(timezone.utc).isoformat()),
        ("normalization_pipeline", OrderedDict([
            ("step_1", "For features with is_log1p=true: x = log1p(x)"),
            ("step_2", "For all 400 scaled features: x = (x - median) / scale"),
            ("step_3", "The 6 mask columns (mask_has_*) are fixed flags (1.0 or 0.0) and are NOT scaled."),
        ])),
        ("total_scaled_features", 400),
        ("total_unscaled_mask_features", 6),
        ("total_lstm_input_features", 406),
        ("unscaled_mask_columns", expected_masks),
    ])

    scaler_doc["features"] = OrderedDict()
    for feature_name in scaler_features:
        params = scaler_raw[feature_name]
        scaler_doc["features"][feature_name] = OrderedDict([
            ("median", params["median"]),
            ("scale", params["scale"]),
            ("is_log1p", params.get("is_log1p", False)),
            ("q25", params.get("q25")),
            ("q75", params.get("q75")),
        ])

    def represent_ordereddict(dumper, data):
        return dumper.represent_mapping("tag:yaml.org,2002:map", data.items())

    yaml.add_representer(OrderedDict, represent_ordereddict)

    OUTPUT_SCALER.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_SCALER, "w", encoding="utf-8") as f:
        yaml.dump(dict(scaler_doc), f, default_flow_style=False, sort_keys=False, allow_unicode=True, width=120)

    # Build inference_feature_order_v2.json
    print("Writing inference_feature_order_v2.json...")
    feature_doc = OrderedDict([
        ("_metadata", OrderedDict([
            ("version", "v2"),
            ("artifact_type", "inference_feature_order"),
            ("description", (
                "Exact ordered list of 406 feature names expected by the LSTM v2 input layer."
            )),
            ("checkpoint", "gaussian_next_state_best_v2.pt"),
            ("checkpoint_commit", meta.get("repository_commit", "unknown")),
            ("exported_at", datetime.now(timezone.utc).isoformat()),
            ("total_features", 406),
            ("input_dimension", 406),
            ("state_dimension", meta.get("state_dimension", 406)),
        ])),
        ("features", meta_features),
    ])

    with open(OUTPUT_FEATURES, "w", encoding="utf-8") as f:
        json.dump(feature_doc, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print("\n[SUCCESS] Contract v2 exported successfully.")
    print(f"  - {OUTPUT_SCALER.relative_to(REPO_ROOT)}")
    print(f"  - {OUTPUT_FEATURES.relative_to(REPO_ROOT)}")

if __name__ == "__main__":
    main()
