#!/bin/bash
set -e

# Redirect all output (stdout and stderr) to ARG_NUMBER-log.log
# Delay setting ARG_NUMBER until we get it from the input
if [ $# -ne 1 ]; then
    echo "Usage: $0 <number>"
    exit 1
fi

ARG_NUMBER="$1"

# Now that ARG_NUMBER is known, set up logging with timestamp
TIMESTAMP=$(date '+%Y-%m-%d_%H-%M-%S')
LOG_FILE="outputs/pod-${ARG_NUMBER}-log-${TIMESTAMP}.log"
exec > >(tee "$LOG_FILE") 2>&1

echo "Received argument: $ARG_NUMBER"

# Generate list based on ARG_NUMBER (1–5)
echo "Generating variant list group $ARG_NUMBER..."
variant_list=()

START=$(( (ARG_NUMBER - 1) * 6 + 1 ))
END=$(( ARG_NUMBER * 6 ))

for num in $(seq $START $END); do
  for v in {1..10}; do
    variant_list+=("${num}_v${v}")
  done
done

# Print the generated list (for verification/logging)
echo "Generated variants:"
printf '%s\n' "${variant_list[@]}"

# Hardcoded HF_TOKEN
HF_TOKEN="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"

# Step 1: Conda env creation (only if it doesn’t exist)
echo "Checking if conda environment 'mimicmotion' exists..."
cd /storage/MimicMotion
if conda env list | grep -qE '^\s*mimicmotion\s'; then
    echo "Conda environment 'mimicmotion' already exists. Skipping creation."
else
    echo "Creating conda environment..."
    conda env create -f environment.yaml
fi

# Initialize conda in the shell
source /opt/miniconda/etc/profile.d/conda.sh
conda init bash

# Activation command string
ENV_ACTIVATION="source /opt/miniconda/etc/profile.d/conda.sh && conda activate mimicmotion"

# Step 4: Install sed if needed
echo "Checking and installing sed if necessary..."
apt-get update && apt-get install -y sed

# Step 5: Patch dynamic_modules_utils.py
echo "Patching dynamic_modules_utils.py..."
PATCH_FILE="/opt/miniconda/envs/mimicmotion/lib/python3.11/site-packages/diffusers/utils/dynamic_modules_utils.py"
sed -i 's/\bcached_download,//g' "$PATCH_FILE"

# Step 6: Hugging Face login
echo "Logging into Hugging Face..."
eval "$ENV_ACTIVATION && huggingface-cli login --token $HF_TOKEN"

echo "Setup complete!"

# Step 7: Run inference for each variant sequentially
echo "Starting inference for each variant..."

for variant in "${variant_list[@]}"; do
    CONFIG_PATH="configs/config-${variant}.yml"
    echo "Running inference with config: $CONFIG_PATH"
    eval "$ENV_ACTIVATION && python inference.py --inference_config $CONFIG_PATH"

    if [ $? -ne 0 ]; then
        echo "Error: Inference failed for variant $variant. Exiting..."
        exit 1
    fi
    echo "Finished inference for $variant"
done

echo "All inference runs completed successfully!"
