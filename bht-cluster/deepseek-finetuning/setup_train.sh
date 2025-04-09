#!/bin/bash

# Step 1: Apply the pod configuration from the YAML file.
echo "Applying deepseek-finetune-pod.yml ..."
kubectl apply -f deepseek-finetune-pod.yml

# Wait for the pod to be running
echo "Waiting for pod 'deepseek-finetune' in namespace 's85468' to be Running..."
while true; do
  status=$(kubectl get pod deepseek-finetune -n s85468 -o jsonpath='{.status.phase}')
  if [[ "$status" == "Running" ]]; then
    echo "Pod is Running!"
    break
  fi
  echo "Current status: $status. Waiting..."
  sleep 2  # Adjust the sleep duration as needed
done

# Step 2 & 3: Install dependencies, change directory, and start an interactive bash session
kubectl -n s85468 exec -it deepseek-finetune -- bash -c "
  apt-get update && \
  apt-get install -y build-essential nano libjpeg-dev libpng-dev && \
  pip install pandas datasets scikit-learn trl transformers sacrebleu unsloth && \
  export CC=/usr/bin/gcc && \
  cd /storage/text2gloss-finetune && \
  python train.py"
