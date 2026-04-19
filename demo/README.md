# Demo

This directory contains runnable examples for the `mpdb` package.

## `demo_mpdb.py`

The main demo script supports three modes:

```bash
python demo/demo_mpdb.py --mode single
python demo/demo_mpdb.py --mode file
python demo/demo_mpdb.py --mode torch
```

- `single`: single-process debugging with `not_distributed=True`
- `file`: multi-process debugging using file-based synchronization
- `torch`: multi-process debugging using `torch.distributed`

## `launch_mpdb.sh`

This helper script uses `accelerate launch` to run the `torch` demo and pipes a few commands into the debugger:

```bash
bash demo/launch_mpdb.sh
```

Notes:

- `torch` mode requires PyTorch.
- `launch_mpdb.sh` also requires `accelerate`.
