#!/bin/bash
cd ~
cd LLaVA-Med

# Start controller
echo "Starting controller..."
python -m llava.serve.controller --host 0.0.0.0 --port 10003 > /dev/null 2>&1 &
controller_pid=$!

# Wait for controller to start
sleep 10  # Adjust this sleep time 

# Start model worker
echo "Starting model worker..."
python -m llava.serve.model_worker --host 0.0.0.0 --controller http://localhost:10003 --port 40000 --worker http://localhost:40000 --model-path /scratch/mahirj.scee.iitmandi/LLaVA-Med-7B --multi-modal --num-gpus 2 > ./output2.txt 2>&1 &
worker_pid=$!

sleep 30  # Adjust this sleep time 

# Execute the test message command and redirect output to output.txt
echo "Test Message..."
python -m llava.serve.test_message --model-name LLaVA-Med-7B --controller http://localhost:10003

# Wait for the processes to be killed
wait $controller_pid
wait $worker_pid

echo "All processes have finished."
