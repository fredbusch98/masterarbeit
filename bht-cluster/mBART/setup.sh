#!/bin/bash

# Step 1: Apply the pod configuration from the YAML file.
echo "Applying mbart-finetune-pod.yml ..."
kubectl apply -f mbart-finetune-pod.yml

# Wait for the pod to be Running
echo "Waiting for pod 'mbart-finetune' in namespace 's85468' to be Running..."
while true; do
  status=$(kubectl get pod mbart-finetune -n s85468 -o jsonpath='{.status.phase}')
  if [[ "$status" == "Running" ]]; then
    echo "Pod is Running!"
    break
  fi
  echo "Current status: $status. Waiting..."
  sleep 2  # Adjust the sleep duration as needed
done

# Step 2 & 3: Install dependencies, upgrade torchvision, change directory, and start an interactive bash session
kubectl -n s85468 exec -it mbart-finetune -- bash -c "
  apt-get update && \
  apt-get install -y build-essential nano libjpeg-dev libpng-dev && \
  pip install --upgrade --no-cache-dir torchvision && \
  pip install pandas sentencepiece torch transformers sacrebleu tqdm accelerate>=0.26.0 && \
  export CC=/usr/bin/gcc && \
  cd /storage/text2gloss-finetune/mBART && \
  exec bash"
