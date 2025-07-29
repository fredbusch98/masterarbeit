#!/bin/bash
set -e

# Load environment variables from .env if it exists
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Check if HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
    echo "Error: HF_TOKEN is not set. Please add it to the .env file."
    exit 1
fi

# Start the pod
echo "Applying mimicmotion-pod.yml ..."
kubectl apply -f mimicmotion-pod.yml

# Wait for the pod to be running
echo "Waiting for pod 'mimicmotion' in namespace 's85468' to be Running..."
while true; do
  status=$(kubectl get pod mimicmotion -n s85468 -o jsonpath='{.status.phase}')
  if [[ "$status" == "Running" ]]; then
    echo "Pod is Running!"
    break
  fi
  echo "Current status: $status. Waiting..."
  sleep 2  # Adjust the sleep duration as needed
done

# Define a helper function to execute commands inside the pod.
exec_in_pod() {
  kubectl exec -n s85468 mimicmotion -- bash -c "$1"
}

# Step 1 & 2 & 3: Create the conda environment.
# Note: Because “conda activate” only affects the current shell,
# we chain commands so that we have access to the new environment.
echo "Creating conda environment..."
exec_in_pod "cd storage/MimicMotion && conda env create -f environment.yaml && \
              source /opt/miniconda/etc/profile.d/conda.sh && conda init bash"

# Activate the environment for a given command. (Could be used to later automatically activate the environment and pass the pose-sequence output of the gloss2pose module?!)
ENV_ACTIVATION="source /opt/miniconda/etc/profile.d/conda.sh && conda activate mimicmotion"

# Ensure sed is installed (with retries in case the NVIDIA mirror is mid‑sync)
echo "Checking and installing sed (with retries) if necessary..."
exec_in_pod "\
  apt-get clean && rm -rf /var/lib/apt/lists/* ; \
  n=0; until [ \$n -ge 5 ]; do \
    apt-get update && break; \
    n=\$((n+1)); \
    echo \"apt-get update failed (attempt \$n), retrying in 5s...\"; \
    sleep 5; \
  done ; \
  apt-get install -y sed"

# Step 4: Patch dynamic_modules_utils.py by removing any line containing 'cached_download'
echo "Patching dynamic_modules_utils.py..."
FILE_TO_PATCH="/opt/miniconda/envs/mimicmotion/lib/python3.11/site-packages/diffusers/utils/dynamic_modules_utils.py"
exec_in_pod "sed -i 's/\bcached_download,//g' ${FILE_TO_PATCH}"

# Step 5: Login to Hugging Face non-interactively.
echo "Logging into Hugging Face..."
exec_in_pod "$ENV_ACTIVATION && huggingface-cli login --token $HF_TOKEN"

echo "Setup complete!"
