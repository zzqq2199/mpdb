# mpdb

`mpdb` (Multiprocess PDB) is a lightweight distributed debugger built on top of `ipdb`.

Repository: <https://github.com/zzqq2199/mpdb>

> **Note:** This project was formerly known as `dpdb`. The `dpdb` package name on PyPI was already taken, so it has been renamed to `mpdb`. Both `import mpdb` and `import dpdb` are supported with identical functionality.

## Release Notes
- 1.2.0: add `set_description()` API for customizing web UI header.
  - New `mpdb.web_pdb.set_description(title, subtitle)` function to override the default "MPDB Web Debugger" title and "by zhouquan" subtitle at runtime.
  - The `/status` API now includes `title` and `subtitle` fields when set.
  - The web UI dynamically updates the header text from the backend status response.
  - `dpdb` package calls `set_description("DPDB Web Debugger")` automatically on import.
- 1.1.2: update the README documentation.
  - Add the repository link for easier source code access.
  - Clarify installation options: PyPI as recommended, GitHub as optional.
- 1.1.1: rename internal identifiers and assets to `mpdb`.
  - All internal identifiers (`dpdb_rank` → `rank`, `dpdb_world_size` → `world_size`) renamed for consistency.
  - Demo files and test files renamed to `mpdb`.
  - `localStorage` keys in web UI updated to `mpdb_*` prefix.
  - Prompt updated to `[mpdb]>>>`.
- 1.1.0: rename from `dpdb` to `mpdb`.
  - PyPI package name `dpdb` was already taken, renamed to `mpdb` (Multiprocess PDB).
  - Full backward compatibility: both `import mpdb` and `import dpdb` work identically.
  - `pip install mpdb` is the recommended installation method.
- 1.0.0: first public release version.
  - single-process debugging
  - distributed command broadcast
  - file-based synchronization for non-`torch.distributed` scenarios
  - a built-in web UI
  - switching between web and terminal modes during a session

## Install

Recommended (PyPI):

```bash
pip install mpdb
```

Optional (GitHub):

```bash
pip install git+https://github.com/zzqq2199/mpdb.git
```

## Quick Start

```python
import mpdb

def train():
    for step in range(10):
        if step == 5:
            mpdb.set_trace()
```

Run a script directly:

```bash
python -m mpdb your_script.py
```

## Demo

The standalone repository ships with runnable examples under `demo/`:

```bash
python demo/demo_mpdb.py --mode single
python demo/demo_mpdb.py --mode file
python demo/demo_mpdb.py --mode torch
```

There is also a helper launcher for the `torch` demo:

```bash
bash demo/launch_mpdb.sh
```

## Notes

- `torch` is optional and only required when you use `sync_method='torch'`.
- The web UI template is packaged with the wheel and works with `pip install mpdb`.
- Both `import mpdb` and `import dpdb` are supported and behave identically.
