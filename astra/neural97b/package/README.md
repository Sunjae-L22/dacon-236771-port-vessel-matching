# Neural97b training-length experiment

This package learns a name-only character/Jamo cross-encoder from scratch. It is
an experiment, not a competition submission. No external data or weights are used.

Run on an allocated GPU:

```sh
python train.py --data data --config config.json --out /path/to/persistent/runNN97 --device cuda
```

The fixed configuration runs 20,000 steps, saves checkpoints every 2,000 steps, and
selects the earliest checkpoint with the highest fold4 conditional name score.
That metric includes the correct name among hard negatives and is not end-to-end
matching accuracy. Training reads neither `outer2.json` nor outer2 Korean labels.
The actual matched79 comparison must follow separately on its held-out candidates.

The supervised fit includes every native Korean row and every LLM row passing
rescoring using only folds0/1/3; fold4 tunes and fold2 is the final holdout. All
catalog English names may be used as unlabeled data, with a fixed sample of4,096
English-name self-supervised examples. Source sampling is40%native/40%LLM/20%catalog,
with shuffled cycles over vessels within each source. Partial-name examples have
weight0.2 and compatible same-prefix/suffix negatives are masked. Random rename
examples are excluded because a name-only model cannot recover their true vessel.

Reference inference:

```sh
python evaluate.py --checkpoint /path/to/runNN97/best.pt --data data \
  --candidates matched79_outer2.jsonl --split outer2 --weight 0.25 --top-k 100 \
  --out /path/to/runNN97/outer2_fixed_weight.json --device cuda
```

Each candidate JSONL row contains `q`, `candidates` (catalog row indices),
`baseline_scores`, `t` (target row index), `sample_id`, and `variant`. Candidate
indices must use the exact packaged catalog ordering. The baseline must exclude
the evaluation ship from its supervised fit. Do not tune the fusion weight on
outer2. Weight0.25 with top100 is a predeclared diagnostic unless a genuinely disjoint fold4
baseline is available. A fold2 baseline trained on fold4 cannot provide fold4
fusion-selection scores.

For a small CPU compatibility test only:

```sh
python train.py --data data --config config.json --out /tmp/nn97_smoke --device cpu --smoke
```

Long CPU training is deliberately disabled. `complete.json`, `history.json`,
`run_config.json`, `best.pt`, and step checkpoints are the persistent outputs.
The parent Colab wrapper must verify/copy outputs to the allocated Drive folder,
flush and sync, and then release its runtime, including when an exception occurs.
This package does not manipulate other runtimes.

This repeats research evaluation on outerfold2 after Neural97. It is not a new
independent holdout. Any adoption requires another untouched validation cohort.
Only total training length and checkpoint interval differ from Neural97; data,
architecture, seed, optimizer, warmup, and corruption recipe are identical.
The existing cosine schedule is stretched to the new training length.
