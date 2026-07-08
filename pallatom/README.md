# pallatom

Training and sampling entry points for the `MainTrunk` all-atom diffusion
model. See the repo root [README.md](../README.md) for what pallatom is and
how it relates to AlphaFold3, and [tests/README.md](tests/README.md) /
[../docs/best_practices.md](../docs/best_practices.md) for code conventions.

There are two ways to train: interactively from `notebook.ipynb` (single
process, good for iterating on a toy checkpoint), or via
`scripts/run_ddp_train.sh` (multi-GPU DDP, for a real training run). Both
ultimately invoke `train/train_loop.py`; sampling is always done through
`sample/sampling.py`.

## Training from `notebook.ipynb`

`notebook.ipynb` (in this directory) walks through the full loop —
configure, train, sample, visualise — against the toy checkpoint
[`pallatom_toy_best.pt`](../pallatom_toy_best.pt) at the repo root:

1. **Configure** (cells 5–6) — build a `TrainConfig` (from
   `train.train_config`) in Python, e.g.:

   ```python
   tcfg = TrainConfig(
       training=TrainingParams(
           num_epochs=50,
           pretrained_weights="pallatom_toy_best.pt",
           resume_checkpoint="pallatom_toy_best.pt",
       ),
       checkpoint=CheckpointParams(checkpoint_path="pallatom_toy_best.pt"),
       train_loader=TrainLoaderConfig(max_seq_length=128),
   )
   ```

   and serialize it to `train/run_config.json` via
   `tcfg.model_dump_json()`.

2. **Train** (cells 7–8) — launch `train/train_loop.py` as a subprocess on a
   background thread, either single-process:

   ```python
   [sys.executable, "-u", "train/train_loop.py",
    "--dataset_jsonl", "data/chain_set.jsonl",
    "--keys_for_splits_json", "data/chain_set_splits.json",
    "--config", "train/run_config.json",
    "--structlog_jsonl", "train/train_logs.jsonl",
    "--shard_dir", "data/shards"]
   ```

   or under `torchrun` with `--ddp` appended, using
   `torch.cuda.device_count()` GPUs (see cell 8). Progress streams to
   `train/train_logs.jsonl` (or `train_logs_ddp.jsonl` for the DDP path) as
   structured JSON lines, and to the subprocess's stdout.

3. **Sample** (cells 11–13) — build a `SampleConfig` reusing the trained
   model/noise params from `tcfg`, pointing `checkpoint.checkpoint_path` at
   the checkpoint written by training, and serialize it to
   `train/sample_config.json`. Then launch `sample/sampling.py` the same way
   as training:

   ```python
   [sys.executable, "-u", "sample/sampling.py",
    "--config", "train/sample_config.json",
    "--log_file", "train/sample_logs.jsonl"]
   ```

   This writes sampled structures as a JSON list of PDB strings to
   `train/samples.json`.

4. **Visualise** (cells 14–15) — load `train/samples.json` and render a
   sampled structure with `py3Dmol`:

   ```python
   with open("train/samples.json") as file:
       data = json.load(file)
   pdb_str = data[7]
   view = py3Dmol.view(width=600, height=600)
   view.addModelsAsFrames(pdb_str)
   view.setStyle({"model": -1}, {"cartoon": {"color": "spectrum"}})
   view.zoomTo()
   view.render()
   ```

The notebook is meant for quick iteration against a small toy checkpoint —
for a real multi-GPU run, use `scripts/run_ddp_train.sh` instead.

## Training with `scripts/run_ddp_train.sh`

[`scripts/run_ddp_train.sh`](../scripts/run_ddp_train.sh) launches a
detached, multi-GPU DDP training run of `train/train_loop.py` under
`torchrun`, backgrounded with `nohup` so it survives an SSH disconnect:

```bash
scripts/run_ddp_train.sh
```

What it does:

1. Detects GPU count via `nvidia-smi` and sets `--nproc_per_node` to match.
2. Runs a 30-second NCCL preflight (`torchrun` + a trivial `all_reduce`) to
   pick the fastest working transport for the current GPU/PCIe topology,
   falling back from P2P → SHM → plain sockets as each fails. This guards
   against a cloud box coming back up post-reboot with a different
   underlying topology, where NCCL's fallback can otherwise crash instead of
   degrading gracefully.
3. Launches training under `torchrun --nproc_per_node=<N> train/train_loop.py`
   with `--ddp`, reading `train/run_config.json` and writing to
   `train/train_logs_ddp.jsonl`.

It expects `data/chain_set.jsonl`, `data/chain_set_splits.json`, and
`train/run_config.json` to already exist — write `run_config.json` the same
way the notebook does (serialize a `TrainConfig`) before running the script.

Useful commands once it's running:

```bash
# Follow stdout
tail -f scripts/train_stdout.log

# Follow structured training metrics
tail -f train/train_logs_ddp.jsonl

# Stop a run started by the script
kill <Training PID printed by the script>
```

## Sampling

Once you have a checkpoint (from either training path above), sample from
it directly with `sample/sampling.py`:

```bash
python -m sample.sampling --config train/sample_config.json --log_file train/sample_logs.jsonl
```

`sample_config.json` must validate against `SampleConfig`
(`sample/sample_config.py`) — see [sample/README.md](sample/README.md) for
the full sampling algorithm, config schema, and output format (a JSON list
of PDB strings).
