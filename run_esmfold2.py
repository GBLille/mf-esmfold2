#!/usr/bin/env python
"""Run ESMFold2 structure predictions on a batch of seeds.

The input is a json request shaped like an AlphaFold3 one:
{
  "name": "H1140",
  "sequences": [
    {"protein": {"id": "A", "sequence": "MKV...", "modifications": [{"ptmType": "SEP", "ptmPosition": 4}]}},
    {"dna": {"id": ["B", "C"], "sequence": "GATC..."}},
    {"ligand": {"id": "L", "ccdCodes": ["SAH"]}},
    {"ligand": {"id": "M", "smiles": "CCO"}}
  ]
}

Each seed produces --num_diffusion_samples predictions, written in --output_dir as
esmf2_seed_<seed>_sample_<sample>.{cif,pkl} and esmf2_seed_<seed>_sample_<sample>_summary_confidences.json.

MSAs computed by AFmassive, ColabFold or AlphaFold3 can be given with --msa_source and --msa_dir.
"""
import argparse
import io
import json
import logging
import os
import pickle
import random
import re
import sys
import time

import numpy as np

ENTITY_TYPES = ("protein", "dna", "rna", "ligand")
MSA_SOURCES = ("none", "afmassive", "colabfold", "alphafold3")

# ==========================================
# 0. ARGUMENTS
# ==========================================
parser = argparse.ArgumentParser(description="Run ESMFold2 predictions on a batch of seeds")
parser.add_argument("--json_path", type=str, required=True, help="Path to the json request (see the module docstring)")
parser.add_argument("--output_dir", type=str, required=True, help="Directory where the predictions are written")
parser.add_argument("--seeds", type=str, default="", help="Comma-separated list of seeds, overrides --random_seed and --num_seeds")
parser.add_argument("--random_seed", type=int, default=None, help="First seed of the batch, random if not given")
parser.add_argument("--num_seeds", type=int, default=1, help="Number of consecutive seeds from --random_seed")
parser.add_argument("--num_diffusion_samples", type=int, default=5, help="Number of diffusion samples per seed")
parser.add_argument("--num_loops", type=int, default=10, help="Number of loops (recycles)")
parser.add_argument("--num_sampling_steps", type=int, default=100, help="Number of diffusion sampling steps")
parser.add_argument("--msa_source", type=str, choices=MSA_SOURCES, default="none",
                    help="Tool that computed the MSAs found in --msa_dir, 'none' to fold without alignments")
parser.add_argument("--msa_dir", type=str, default="",
                    help="AFmassive 'msas' directory, ColabFold 'msas_colabfold' directory or AlphaFold3"
                    " 'msas_alphafold3' directory (or its msas_alphafold3_data.json file)")
parser.add_argument("--max_msa_sequences", type=int, default=200000, help="Maximum number of sequences read per MSA file")
parser.add_argument("--msa_max_depth", type=int, default=1024, help="Maximum MSA depth used by the model")
parser.add_argument("--model", type=str, default="biohub/ESMFold2", help="Hugging Face name or local path of the model")
parser.add_argument("--device", type=str, default="cuda", help="Torch device used for the predictions")

# ==========================================
# 1. INPUT
# ==========================================
def load_request(json_path):
    with open(json_path, "r") as f:
        request = json.load(f)
    if not request.get("sequences"):
        raise ValueError(f"No 'sequences' in {json_path}")
    for entry in request["sequences"]:
        if len(entry) != 1 or next(iter(entry)) not in ENTITY_TYPES:
            raise ValueError(f"Each sequence should be one of {'|'.join(ENTITY_TYPES)}, got: {entry}")
    return request

def entity_of(entry):
    entity = next(iter(entry))
    return entity, entry[entity]

def chain_ids(content):
    ids = content["id"]
    return [ids] if isinstance(ids, str) else list(ids)

def protein_sequences(request):
    """Unique protein sequences of the request, in order."""
    sequences = []
    for entry in request["sequences"]:
        entity, content = entity_of(entry)
        if entity == "protein" and content["sequence"] not in sequences:
            sequences.append(content["sequence"])
    return sequences

def to_modifications(entity, content):
    from esm.models.esmfold2 import Modification
    modifications = []
    # AlphaFold3 fields, positions starting at 1
    type_key, position_key = ("ptmType", "ptmPosition") if entity == "protein" else ("modificationType", "basePosition")
    for modification in content.get("modifications") or []:
        modifications.append(Modification(position=int(modification[position_key]) - 1, ccd=modification[type_key]))
    return modifications or None

def build_input(request, msas):
    from esm.models.esmfold2 import DNAInput, LigandInput, ProteinInput, RNAInput, StructurePredictionInput
    type_to_class = {"protein": ProteinInput, "dna": DNAInput, "rna": RNAInput}

    chain_inputs = []
    for entry in request["sequences"]:
        entity, content = entity_of(entry)
        ids = content["id"]
        if entity == "ligand":
            if content.get("ccdCodes"):
                chain_inputs.append(LigandInput(id=ids, ccd=list(content["ccdCodes"])))
            elif content.get("smiles"):
                chain_inputs.append(LigandInput(id=ids, smiles=content["smiles"]))
            else:
                raise ValueError(f"Ligand {ids} needs 'ccdCodes' or 'smiles'")
            continue

        sequence = content["sequence"]
        kwargs = {"id": ids, "sequence": sequence, "modifications": to_modifications(entity, content)}
        if entity == "protein" and msas.get(sequence) is not None:
            kwargs["msa"] = msas[sequence]
        chain_inputs.append(type_to_class[entity](**kwargs))
    return StructurePredictionInput(sequences=chain_inputs)

def resolve_seeds(args):
    if args.seeds:
        return [int(seed) for seed in args.seeds.split(",") if seed.strip()]
    first_seed = args.random_seed if args.random_seed is not None else random.randint(0, 4_000_000_000)
    return [first_seed + i for i in range(args.num_seeds)]

# ==========================================
# 2. MSAS
# ==========================================
# rows sharing a 'key=N' header token are paired across chains by ESMFold2 (esm/models/esmfold2/paired_msa.py)
KEY_PATTERN = re.compile(r"key=(-?\d+)")
# same species extraction as AlphaFold2 multimer (alphafold/data/msa_identifiers.py)
UNIPROT_PATTERN = re.compile(
    r"^(?:tr|sp)\|(?:[A-Z0-9]{6,10})(?:_\d+)?\|(?:[A-Z0-9]{1,10}_)(?P<SpeciesIdentifier>[A-Z0-9]{1,5})")

def with_headers(msa, headers):
    from esm.utils.msa.msa import MSA
    from esm.utils.parsing import FastaEntry
    entries = [ FastaEntry(header, entry.sequence) for header, entry in zip(headers, msa.entries) ]
    return MSA(entries, deletions=msa.deletions)

def unpaired(msa):
    """Remove any pairing key from the headers so that the rows stay unpaired."""
    return with_headers(msa, [ KEY_PATTERN.sub("nokey", entry.header) for entry in msa.entries ])

def stack(msas):
    from esm.utils.msa.msa import MSA
    msas = [ msa for msa in msas if msa is not None and msa.depth > 0 ]
    if not msas:
        return None
    return MSA.stack(msas) if len(msas) > 1 else msas[0]

def afmassive_msa_dirs(msa_dir, sequences):
    """Map each protein sequence to the AFmassive directory holding its alignments."""
    chain_id_map_path = os.path.join(msa_dir, "chain_id_map.json")
    sequence_dirs = {}
    if os.path.isfile(chain_id_map_path):
        # multimer: one directory per unique sequence, named after its first chain
        with open(chain_id_map_path, "r") as f:
            chain_id_map = json.load(f)
        for chain_id, chain in chain_id_map.items():
            chain_dir = os.path.join(msa_dir, chain_id)
            if chain["sequence"] not in sequence_dirs and os.path.isdir(chain_dir):
                sequence_dirs[chain["sequence"]] = chain_dir
    elif len(sequences) == 1:
        # monomer: the alignments are directly in the msas directory
        sequence_dirs[sequences[0]] = msa_dir
    return sequence_dirs

def species_keyed(msa, species_keys):
    headers = []
    for i, entry in enumerate(msa.entries):
        match = UNIPROT_PATTERN.match(entry.header.strip())
        if i == 0 or not match:
            headers.append(KEY_PATTERN.sub("nokey", entry.header))
            continue
        species = match.group("SpeciesIdentifier")
        key = species_keys.setdefault(species, len(species_keys) + 1)
        headers.append(f"{KEY_PATTERN.sub('nokey', entry.header)} key={key}")
    return with_headers(msa, headers)

def read_afmassive_msas(msa_dir, sequences, max_sequences):
    from esm.utils.msa.msa import MSA
    pairing = len(sequences) > 1
    species_keys = {}
    msas = {}
    for sequence, chain_dir in afmassive_msa_dirs(msa_dir, sequences).items():
        if sequence not in sequences:
            continue
        path = lambda name: os.path.join(chain_dir, name)
        parts = []
        if os.path.isfile(path("bfd_uniref_hits.a3m")):
            parts.append(MSA.from_a3m(path("bfd_uniref_hits.a3m"), remove_insertions=True, max_sequences=max_sequences))
        elif os.path.isfile(path("small_bfd_hits.sto")):
            parts.append(MSA.from_stockholm(path("small_bfd_hits.sto"), remove_insertions=True, max_sequences=max_sequences))
        for name in ("mgnify_hits.sto", "uniref90_hits.sto"):
            if os.path.isfile(path(name)):
                parts.append(MSA.from_stockholm(path(name), remove_insertions=True, max_sequences=max_sequences))
        parts = [ unpaired(part) for part in parts ]
        if pairing and os.path.isfile(path("uniprot_hits.sto")):
            uniprot = MSA.from_stockholm(path("uniprot_hits.sto"), remove_insertions=True, max_sequences=max_sequences)
            parts.append(species_keyed(uniprot, species_keys))
        msas[sequence] = stack(parts)
    return msas

def read_msas(source, msa_dir, sequences, max_sequences):
    """Return {protein sequence: MSA} for the sequences found in the alignments of the source."""
    if source == "none" or not sequences:
        return {}
    if not os.path.exists(msa_dir):
        raise FileNotFoundError(f"MSA source '{source}' given but --msa_dir '{msa_dir}' does not exist")
    readers = {
        "afmassive": read_afmassive_msas,
    }
    msas = readers[source](msa_dir, sequences, max_sequences)
    checked = {}
    for sequence, msa in msas.items():
        if msa is None:
            continue
        if msa.query != sequence:
            logging.warning(f"MSA query differs from the sequence, MSA ignored:\n{msa.query}\n{sequence}")
            continue
        checked[sequence] = msa
    return checked

# ==========================================
# 3. OUTPUT
# ==========================================
def add_atom_site_occupancy(mmcif_text, occupancy="1.00"):
    """Add _atom_site.occupancy (needed by Biopython's MMCIFParser) to the atom rows when it is missing."""
    lines = mmcif_text.splitlines()
    output_lines = []
    i = 0
    while i < len(lines):
        if lines[i].strip() != "loop_":
            output_lines.append(lines[i])
            i += 1
            continue
        output_lines.append(lines[i])
        i += 1
        tags = []
        while i < len(lines) and lines[i].strip().startswith("_"):
            tags.append(lines[i].strip().split()[0])
            output_lines.append(lines[i])
            i += 1
        if not any(tag.startswith("_atom_site.") for tag in tags) or "_atom_site.occupancy" in tags:
            continue
        output_lines.append("_atom_site.occupancy")
        while i < len(lines):
            stripped = lines[i].strip()
            if stripped in ("", "#", "loop_") or stripped.startswith(("data_", "save_", "_")):
                break
            output_lines.append(f"{lines[i]} {occupancy}")
            i += 1
    return "\n".join(output_lines) + "\n"

def write_sample(result, seed, sample, output_dir, chains):
    name = f"esmf2_seed_{seed}_sample_{sample}"

    mmcif = result.complex.to_mmcif()
    if "_atom_site.occupancy" not in mmcif:
        mmcif = add_atom_site_occupancy(mmcif)
    with open(os.path.join(output_dir, f"{name}.cif"), "w") as f:
        f.write(mmcif)

    plddt = result.plddt.float().numpy() * 100
    pae = result.pae.float().numpy()
    with open(os.path.join(output_dir, f"{name}.pkl"), "wb") as f:
        pickle.dump({
            "plddt": plddt,
            "predicted_aligned_error": pae,
            "max_predicted_aligned_error": float(np.max(pae)),
        }, f)

    summary = {
        "seed": seed,
        "sample": sample,
        "plddt": float(plddt.mean()),
        "ptm": float(result.ptm),
        "iptm": float(result.iptm),
        "chain_ids": chains,
    }
    if result.pair_chains_iptm is not None:
        summary["chain_pair_iptm"] = [ [ round(float(v), 4) for v in row ] for row in result.pair_chains_iptm.float().tolist() ]
    with open(os.path.join(output_dir, f"{name}_summary_confidences.json"), "w") as f:
        json.dump(summary, f, indent=1)
    logging.info(f"{name}: pLDDT {summary['plddt']:.2f}, pTM {summary['ptm']:.4f}, ipTM {summary['iptm']:.4f}")

def all_chain_ids(request):
    return [ chain for entry in request["sequences"] for chain in chain_ids(entity_of(entry)[1]) ]

# ==========================================
# 4. MAIN
# ==========================================
def main(argv=None):
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    request = load_request(args.json_path)
    seeds = resolve_seeds(args)
    os.makedirs(args.output_dir, exist_ok=True)

    sequences = protein_sequences(request)
    msas = read_msas(args.msa_source, args.msa_dir, sequences, args.max_msa_sequences)
    for sequence in sequences:
        if sequence in msas:
            logging.info(f"MSA of {msas[sequence].depth} sequences for protein {sequence[:20]}...")
        else:
            logging.info(f"No MSA for protein {sequence[:20]}..., folded as single sequence")
    spi = build_input(request, msas)

    from esm.models.esmfold2 import ESMFold2InputBuilder, EsmFold2Model
    logging.info(f"Loading {args.model} on {args.device}")
    model = EsmFold2Model.from_pretrained(args.model, device=args.device).eval()

    chains = all_chain_ids(request)
    logging.info(f"Seeds of the batch: {', '.join(map(str, seeds))}")
    for seed in seeds:
        start = time.time()
        fold_output = ESMFold2InputBuilder().fold(
            model,
            spi,
            num_loops=args.num_loops,
            num_sampling_steps=args.num_sampling_steps,
            num_diffusion_samples=args.num_diffusion_samples,
            msa_max_depth=args.msa_max_depth,
            seed=seed,
        )
        # read by MassiveFold to calibrate the batch size
        logging.info(f"Seed {seed} predict time {time.time() - start:.1f}s")
        results = fold_output if isinstance(fold_output, list) else [fold_output]
        for sample, result in enumerate(results):
            write_sample(result, seed, sample, args.output_dir, chains)
    return 0

if __name__ == "__main__":
    sys.exit(main())
