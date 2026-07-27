# AMP_datasets

PathoMIC / PLM-AMP regression datasets and preprocessing artifacts.

## Layout

- `PathoMIC/` ！ cleaned MIC tables, taxonomy graph, species embeddings, train/val/test splits
- `splits1/` ！ split bundle used by training scripts (`splits2` provided as `splits2.tar.gz`)
- `data_sources/` ！ raw downloads from public AMP/MIC databases
- `scripts/` ！ dataset construction and taxonomy update utilities
- `taxdump.tar.gz` ！ NCBI taxonomy dump (root copy)

## Large files

`PathoMIC/taxdump/names.dmp` and `nodes.dmp` are not tracked (GitHub 100 MB limit). Extract them locally:

```bash
tar -xzf PathoMIC/taxdump/taxdump.tar.gz -C PathoMIC/taxdump
```

## Usage with PLM_AMP_Regression

Typical paths (override with CLI flags as needed):

```bash
--data_path /path/to/AMP_datasets/splits1
--taxo_graph_path /path/to/AMP_datasets/PathoMIC/taxonomy_graph.pt
```

Extract `splits2.tar.gz` before using `--data_path .../splits2`.
