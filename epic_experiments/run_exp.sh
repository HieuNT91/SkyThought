#!/bin/bash
# Define configurations
cot_temperature=("0")
# cot_models=("Qwen/Qwen2.5-Coder-1.5B-Instruct" "Qwen/Qwen2.5-Coder-3B-Instruct" "Qwen/Qwen2.5-Coder-7B-Instruct")
cot_models=("Qwen/Qwen2.5-Coder-1.5B-Instruct" "Qwen/Qwen2.5-Coder-3B-Instruct")
cot_seeds=("0")
# cot_datasets=("livecodebench_easy" "livecodebench_medium" "livecodebench_hard")
cot_datasets=("livecodebench_easy" "livecodebench_medium")

best16_temperature=("0.2" "0.6")
# best16_models=("Qwen/Qwen2.5-Coder-1.5B-Instruct" "Qwen/Qwen2.5-Coder-3B-Instruct" "Qwen/Qwen2.5-Coder-7B-Instruct")
best16_models=("Qwen/Qwen2.5-Coder-1.5B-Instruct" "Qwen/Qwen2.5-Coder-3B-Instruct")
best16_seeds=("0")
# best16_datasets=("livecodebench_easy" "livecodebench_medium" "livecodebench_hard")
best16_datasets=("livecodebench_easy" "livecodebench_medium")

# Base directory for results
base_result_dir="sh_results"
progress_file="progress.txt"

# Ensure progress file exists
touch "$progress_file"

# Helper function to check if a combination is already completed
is_completed() {
  local combo="$1"
  grep -Fxq "$combo" "$progress_file"
}

# Helper function to save progress
mark_completed() {
  local combo="$1"
  echo "$combo" >> "$progress_file"
}

# Function to run a single combination
run_combo() {
  local temperature="$1"
  local model="$2"
  local seed="$3"
  local dataset="$4"
  local method="$5"
  local n="$6"

  local result_subdir="$base_result_dir/$method"
  local result_dir="$result_subdir/$dataset-$model-$seed-$temperature"

  # Skip if already completed
  local combo="$method:$temperature;$model;$seed;$dataset"
  if is_completed "$combo"; then
    echo "Skipping previously completed combo: $combo"
    return
  fi

  # Ensure result directory exists
  mkdir -p "$result_dir"

  # Build and run the command
  echo "Running combination: $combo"
  skythought evaluate \
    --model "$model" \
    --task "$dataset" \
    --seed "$seed" \
    --sampling-params "temperature=$temperature,top_p=1,max_tokens=4096" \
    --n "$n" \
    --result-dir "$result_dir" \
    --system-prompt-name "qwen_cot" \
    --batch-size 32 \
    --backend vllm \
    --backend-args tensor_parallel_size=2,gpu_memory_utilization=0.8

  # Mark the combination as completed
  mark_completed "$combo"
}

# Run all COT combinations
for temperature in "${cot_temperature[@]}"; do
  for model in "${cot_models[@]}"; do
    for seed in "${cot_seeds[@]}"; do
      for dataset in "${cot_datasets[@]}"; do
        run_combo "$temperature" "$model" "$seed" "$dataset" "cot" 1
      done
    done
  done
done

# Run all Best-of-16 combinations
for temperature in "${best16_temperature[@]}"; do
  for model in "${best16_models[@]}"; do
    for seed in "${best16_seeds[@]}"; do
      for dataset in "${best16_datasets[@]}"; do
        run_combo "$temperature" "$model" "$seed" "$dataset" "best_of_16" 16
      done
    done
  done
done