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
INFO:     Started server process [281815]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
Training [1/5]: 100%|██████████████████████| 469/469 [00:15<00:00, 29.85it/s, loss=1.3065]
Evaluating: 100%|█████████████████████████████████████████| 79/79 [00:02<00:00, 38.98it/s]
Epoch 1/5 train_loss=1.242765 test_loss=0.470435 test_acc=85.86%
Training [2/5]: 100%|██████████████████████| 469/469 [00:16<00:00, 28.17it/s, loss=0.4172]
Evaluating: 100%|█████████████████████████████████████████| 79/79 [00:02<00:00, 37.41it/s]
Epoch 2/5 train_loss=0.409984 test_loss=0.329547 test_acc=91.06%
Training [3/5]: 100%|██████████████████████| 469/469 [00:16<00:00, 28.04it/s, loss=0.2934]
Evaluating: 100%|█████████████████████████████████████████| 79/79 [00:02<00:00, 37.04it/s]
Epoch 3/5 train_loss=0.290710 test_loss=0.245883 test_acc=93.00%
Training [4/5]: 100%|██████████████████████| 469/469 [00:17<00:00, 26.99it/s, loss=0.2276]
Evaluating: 100%|█████████████████████████████████████████| 79/79 [00:02<00:00, 35.21it/s]
Epoch 4/5 train_loss=0.224438 test_loss=0.189991 test_acc=94.68%
Training [5/5]: 100%|██████████████████████| 469/469 [00:18<00:00, 25.48it/s, loss=0.1755]
Evaluating: 100%|█████████████████████████████████████████| 79/79 [00:02<00:00, 28.14it/s]
Epoch 5/5 train_loss=0.174463 test_loss=0.146600 test_acc=95.89%
INFO:     Shutting down
INFO:     Waiting for application shutdown.
INFO:     Application shutdown complete.
INFO:     Finished server process [281815]
```
