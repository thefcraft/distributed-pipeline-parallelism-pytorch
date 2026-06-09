# Example (Toy)

```bash
uv run example-cnn/machine_a.py 
INFO:     Started server process [271997]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8001 (Press CTRL+C to quit)
```

```bash
uv run example-cnn/machine_b.py 
INFO:     Started server process [271905]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8002 (Press CTRL+C to quit)
/home/a8c3b742-5df0-8324-baef/local-development/distributed-pipeline-parallelism-pytorch/.venv/lib/python3.13/site-packages/torch/autograd/graph.py:882: UserWarning: Attempting to run cuBLAS, but there was no current CUDA context! Attempting to set the primary context... (Triggered internally at /pytorch/aten/src/ATen/cuda/CublasHandlePool.cpp:370.)
  return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
```

```bash
uv run example-cnn/meta.py 
INFO:     Started server process [272126]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
Training [1/5]: 100%|█████████████████████████| 469/469 [00:17<00:00, 26.59it/s, loss=450.5195]
Evaluating: 100%|██████████████████████████████████████████████| 79/79 [00:02<00:00, 35.27it/s]
Epoch 1/5 train_loss=411.980135 test_loss=29.407473 test_acc=33.49%
Training [2/5]: 100%|██████████████████████████| 469/469 [00:17<00:00, 26.37it/s, loss=73.2120]
Evaluating: 100%|██████████████████████████████████████████████| 79/79 [00:02<00:00, 31.94it/s]
Epoch 2/5 train_loss=127.045119 test_loss=1124.003917 test_acc=10.10%
Training [3/5]: 100%|█████████████████████████| 469/469 [00:17<00:00, 26.12it/s, loss=421.8322]
Evaluating: 100%|██████████████████████████████████████████████| 79/79 [00:02<00:00, 32.63it/s]
Epoch 3/5 train_loss=380.078229 test_loss=5.008089 test_acc=86.00%
Training [4/5]: 100%|███████████████████████████| 469/469 [00:18<00:00, 25.68it/s, loss=5.1623]
Evaluating: 100%|██████████████████████████████████████████████| 79/79 [00:02<00:00, 33.21it/s]
Epoch 4/5 train_loss=5.016060 test_loss=2.191541 test_acc=91.84%
Training [5/5]: 100%|███████████████████████████| 469/469 [00:18<00:00, 25.45it/s, loss=1.2779]
Evaluating: 100%|██████████████████████████████████████████████| 79/79 [00:02<00:00, 32.75it/s]
Epoch 5/5 train_loss=1.229969 test_loss=0.677360 test_acc=92.84%
INFO:     Shutting down
INFO:     Waiting for application shutdown.
INFO:     Application shutdown complete.
INFO:     Finished server process [272126]
```
