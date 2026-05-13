"""
Benchmark Orchestrator — Iterative Hard-Negative Mining Pipeline

Execution flow:
  Round 0: mine (frozen DINOv3) → train V2 for N epochs
  Round 1: remine (trained encoder) → train V2 for N more epochs
  ...
  Round K: remine (trained encoder) → train V2 to completion
  → train V1 (BYOL, unchanged)

Parquet files are versioned: hard_negatives_round_{N}.parquet
"""
import subprocess
import sys
import os
import yaml


def run_command(cmd, description, log_file, max_retries=1, cooldown=30):
    """Run a subprocess with live stdout streaming, logging, and auto-retry."""
    import time as _time
    
    for attempt in range(1, max_retries + 1):
        suffix = f" (attempt {attempt}/{max_retries})" if max_retries > 1 else ""
        msg = f"\n{'='*40}\nStarting: {description}{suffix}\n{'='*40}\n"
        print(msg, end="")
        with open(log_file, "a") as f:
            f.write(msg)

        python_bin = f"{sys.executable} -u"
        full_cmd = f"{python_bin} {cmd}"
        print(f"  CMD: {full_cmd}", flush=True)
        
        process = subprocess.Popen(
            full_cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        with open(log_file, "a") as f:
            for line in process.stdout:
                print(line, end="", flush=True)
                f.write(line)
                f.flush()

        process.wait()

        if process.returncode == 0:
            succ_msg = f"[Success] {description} complete.\n"
            print(succ_msg, end="")
            with open(log_file, "a") as f:
                f.write(succ_msg)
            return True
        
        err_msg = f"[Error] {description} failed (exit code {process.returncode}).\n"
        print(err_msg, end="")
        with open(log_file, "a") as f:
            f.write(err_msg)
        
        if attempt < max_retries:
            retry_msg = f"[Retry] Waiting {cooldown}s before retry {attempt+1}/{max_retries} (will resume from checkpoint)...\n"
            print(retry_msg, end="", flush=True)
            with open(log_file, "a") as f:
                f.write(retry_msg)
            _time.sleep(cooldown)
    
    return False


def main():
    log_file = "benchmark_run.log"
    config_path = "config_v2.yaml"
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    total_epochs = config["training"]["num_epochs"]
    mining_cfg = config.get("mining", {})
    remine_interval = mining_cfg.get("remine_interval", 2)
    num_negatives = mining_cfg.get("negatives_per_query", 4)
    top_k = mining_cfg.get("top_k_candidates", 50)
    safe_radius = mining_cfg.get("safe_radius_meters", 100.0)
    
    msg = f"\nCommencing Iterative Pipeline: {total_epochs} epochs, remine every {remine_interval}\n"
    print(msg, end="")
    with open(log_file, "a") as f:
        f.write(msg)
    
    # Calculate mining rounds
    mining_round = 0
    epoch_cursor = 0
    
    while epoch_cursor < total_epochs:
        next_stop = min(epoch_cursor + remine_interval, total_epochs)
        
        # --- Mining ---
        if mining_round == 0:
            mine_cmd = (
                f"mine_negatives.py --config {config_path} "
                f"--round {mining_round} "
                f"--num-negatives {num_negatives} "
                f"--top-k {top_k} "
                f"--safe-radius {safe_radius}"
            )
            mine_desc = f"Mining Round {mining_round} (frozen DINOv3)"
        else:
            mine_cmd = (
                f"mine_negatives.py --config {config_path} "
                f"--round {mining_round} "
                f"--use-trained "
                f"--num-negatives {num_negatives} "
                f"--top-k {top_k} "
                f"--safe-radius {safe_radius}"
            )
            mine_desc = f"Mining Round {mining_round} (trained encoder)"
        
        mining_ok = True  # assume success unless we need to run
        parquet_file = f"hard_negatives_round_{mining_round}.parquet"
        
        if os.path.exists(parquet_file):
            print(f"[Skip] {parquet_file} already exists — skipping mining round {mining_round}.")
            with open(log_file, "a") as f:
                f.write(f"[Skip] {parquet_file} already exists — skipping mining round {mining_round}.\n")
        else:
            mining_ok = run_command(mine_cmd, mine_desc, log_file, max_retries=3)
            if not mining_ok:
                print(f"Mining round {mining_round} failed — aborting.")
                return
        
        # --- Training ---
        train_cmd = (
            f"train_v2.py --config {config_path} "
            f"--parquet {parquet_file} "
            f"--max-epoch {next_stop}"
        )
        train_desc = f"V2 Training (epochs {epoch_cursor+1}→{next_stop}, round {mining_round})"
        
        train_ok = run_command(train_cmd, train_desc, log_file, max_retries=3)
        if not train_ok:
            print(f"V2 training failed at round {mining_round} after 3 retries — skipping to V1.")
            break
        
        epoch_cursor = next_stop
        mining_round += 1
    
    # --- V1 BYOL Training (isolated, unchanged) ---
    run_command("train_v1.py", "V1 (BYOL)", log_file, max_retries=2)


if __name__ == "__main__":
    main()
