# mf-esmfold2

`run_esmfold2.py` runs [ESMFold2](https://huggingface.co/biohub/ESMFold2) predictions on a batch of seeds. It is the
ESMFold2 engine of [MassiveFold](https://github.com/GBLille/MassiveFold), and can also be used alone.

It reuses the alignments already computed by AFmassive, ColabFold or AlphaFold3, so that the same MSAs can be shared
between engines.

## Installation

```bash
pip install esm==3.4.3
```

`esm` requires python >= 3.12 and torch >= 2.11. The weights (about 27 GB, ESMC 6B included, not gated) are downloaded
from Hugging Face on first use; on a cluster without internet access on the compute nodes, download them beforehand and
point `HF_HOME` to them (with `HF_HUB_OFFLINE=1` in the jobs):

```bash
HF_HOME=<weights_dir> hf download biohub/ESMFold2
```

## Usage

```bash
run_esmfold2.py \
  --json_path request.json \
  --output_dir batch_0 \
  --random_seed 42 --num_seeds 2 \
  --num_diffusion_samples 5 --num_loops 10 --num_sampling_steps 100 \
  --msa_source afmassive --msa_dir output/H1140/msas
```

### Request

The request follows the AlphaFold3 json input, with `protein`, `dna`, `rna` and `ligand` entities:

```json
{
  "name": "H1140",
  "sequences": [
    {"protein": {"id": ["A", "B"], "sequence": "MKV...", "modifications": [{"ptmType": "SEP", "ptmPosition": 4}]}},
    {"dna": {"id": "C", "sequence": "GATC", "modifications": [{"modificationType": "5CM", "basePosition": 2}]}},
    {"ligand": {"id": "L", "ccdCodes": ["SAH"]}},
    {"ligand": {"id": "M", "smiles": "CCO"}}
  ]
}
```

Modification positions start at 1, as in AlphaFold3.

### MSAs

`--msa_source` selects the tool that computed the alignments in `--msa_dir`:

| `--msa_source` | `--msa_dir` | Read |
|---|---|---|
| `none` | | single sequence |
| `afmassive` | AFmassive `msas` directory | `bfd_uniref_hits.a3m` (or `small_bfd_hits.sto`), `mgnify_hits.sto`, `uniref90_hits.sto`; `uniprot_hits.sto` paired by species for multimers |
| `colabfold` | `colabfold_search` output directory (`msas_colabfold`) | the a3m of the job, its paired and unpaired blocks |
| `alphafold3` | `msas_alphafold3` directory or its `msas_alphafold3_data.json` | `pairedMsa` and `unpairedMsa` of each protein |

The alignments are matched to the proteins by sequence. ESMFold2 pairs the rows of different chains sharing a `key=N`
header token: the paired rows of ColabFold and AlphaFold3 are keyed by their row index, the uniprot hits of AFmassive by
their species. A protein without alignment is folded as single sequence. Only protein MSAs are used by ESMFold2.

### Outputs

For each seed `S` and sample `K`, in `--output_dir`:

- `esmf2_seed_S_sample_K.cif`: the structure, pLDDT in the B-factor column,
- `esmf2_seed_S_sample_K.pkl`: `plddt` (0-100), `predicted_aligned_error` and `max_predicted_aligned_error`,
- `esmf2_seed_S_sample_K_summary_confidences.json`: mean `plddt`, `ptm`, `iptm`, `chain_pair_iptm` and `chain_ids`.

The log gives `Seed S predict time <x>s` for each seed.
