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

# rows sharing a 'key=N' header token are paired across chains by ESMFold2 (esm/models/esmfold2/paired_msa.py)
KEY_PATTERN = re.compile(r"key=(-?\d+)")
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
    # same species extraction as AFmassive multimer pairing
    from alphafold.data.msa_identifiers import get_identifiers
    headers = []
    for i, entry in enumerate(msa.entries):
        species = get_identifiers(entry.header).species_id
        if i == 0 or not species:
            headers.append(KEY_PATTERN.sub("nokey", entry.header))
            continue
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

def row_keyed(msa, offset=0):
    """Key each row by its index: row i of each chain is paired with row i of the others."""
    headers = [ entry.header if i == 0 else f"{KEY_PATTERN.sub('nokey', entry.header)} key={offset + i}"
                for i, entry in enumerate(msa.entries) ]
    return with_headers(msa, headers)

def a3m_text_to_msa(a3m_text, max_sequences):
    from esm.utils.msa.msa import MSA
    if not a3m_text or not a3m_text.strip():
        return None
    return MSA.from_a3m(io.StringIO(a3m_text), remove_insertions=True, max_sequences=max_sequences)

def read_alphafold3_msas(msa_dir, sequences, max_sequences):
    """Alignments of the AlphaFold3 data pipeline (msas_alphafold3_data.json)."""
    data_json = msa_dir if os.path.isfile(msa_dir) else os.path.join(msa_dir, "msas_alphafold3_data.json")
    with open(data_json, "r") as f:
        data = json.load(f)

    def msa_text(protein, key):
        # inline a3m, or a path once lightened by MassiveFold
        if protein.get(key):
            return protein[key]
        path = protein.get(f"{key}Path")
        if path and os.path.isfile(path):
            with open(path, "r") as f:
                return f.read()
        return ""

    msas = {}
    for entry in data["sequences"]:
        entity, protein = entity_of(entry)
        if entity != "protein" or protein["sequence"] not in sequences:
            continue
        paired = a3m_text_to_msa(msa_text(protein, "pairedMsa"), max_sequences)
        unpaired_msa = a3m_text_to_msa(msa_text(protein, "unpairedMsa"), max_sequences)
        msas[protein["sequence"]] = stack([
            row_keyed(paired) if paired is not None else None,
            unpaired(unpaired_msa) if unpaired_msa is not None else None,
        ])
    return msas

def normalize_a3m(lines):
    """Copy of colabfold.batch.normalize_a3m, colabfold 1.6.3."""
    out = []
    i = 0
    # keep meta header
    if lines and lines[0].startswith("#"):
        out.append(lines[0].rstrip("\n"))
        i = 1
    header = None
    seq_chunks = []
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                out.append(header)
                out.append("".join(seq_chunks))
            header = line
            seq_chunks = []
        else:
            # remove all whitespace inside sequence lines
            seq_chunks.append("".join(line.split()))
    if header is not None:
        out.append(header)
        out.append("".join(seq_chunks))
    return out

def split_colabfold_a3m(a3m_text):
    """Port of colabfold.batch.unserialize_msa, colabfold 1.6.3.
    A multimer a3m starts with '#len1,len2<TAB>card1,card2', rows with residues in several chains 
    are paired, the others are unpaired rows of their chain. As ESMFold2 folds homo-oligomers from 
    the unpaired MSA, no paired block of query copies is made.
    """
    lines = normalize_a3m(a3m_text.replace("\x00", "").splitlines())
    if not lines[0].startswith("#") or len(lines[0][1:].split("\t")) != 2:
        query = lines[1] if lines[0].startswith(">") else lines[0]
        return {query: ("", "\n".join(lines) + "\n")}

    lengths_field, cardinality_field = lines[0][1:].split("\t")
    lengths = [ int(length) for length in lengths_field.split(",") ]
    cardinalities = [ int(cardinality) for cardinality in cardinality_field.split(",") ]
    num_chains = len(lengths)
    is_single_protein = num_chains == 1 and cardinalities[0] == 1
    is_homooligomer = num_chains == 1 and cardinalities[0] > 1

    queries = []
    start = 0
    for length in lengths:
        queries.append(lines[2][start:start + length])
        start += length

    def split_by_alignment(sequence):
        segments, has_residues = [], []
        start, aligned, residues = 0, 0, False
        for i, residue in enumerate(sequence):
            if residue.islower():
                continue
            residues = residues or (residue != "-" and residue.isupper())
            aligned += 1
            if aligned == lengths[len(segments)]:
                segments.append(sequence[start:i + 1])
                has_residues.append(residues)
                start, aligned, residues = i + 1, 0, False
                if len(segments) == num_chains:
                    break
        return segments, has_residues

    paired = [ [] for _ in range(num_chains) ]
    unpaired_rows = [ [] for _ in range(num_chains) ]
    seen = set()
    for i in range(1, len(lines) - 1, 2):
        header, sequence = lines[i], lines[i + 1]
        if (header, sequence) in seen:
            continue
        seen.add((header, sequence))
        segments, has_residues = split_by_alignment(sequence)
        if len(segments) != num_chains:
            continue
        if not is_single_protein and not is_homooligomer and sum(has_residues) > 1:
            labels = header.lstrip(">").split("\t")
            for j, segment in enumerate(segments):
                paired[j].append(f">{labels[min(j, len(labels) - 1)]}\n{segment}\n")
        else:
            for j, segment in enumerate(segments):
                if has_residues[j]:
                    unpaired_rows[j].append(f"{header}\n{segment}\n")

    chains = {}
    for j, query in enumerate(queries):
        # each chain's MSA starts with its query
        unpaired_text = "".join(unpaired_rows[j])
        if not unpaired_text.startswith(f">") or unpaired_text.split("\n")[1] != query:
            unpaired_text = f">query\n{query}\n" + unpaired_text
        chains[query] = ("".join(paired[j]), unpaired_text)
    return chains

def read_colabfold_msas(msa_dir, sequences, max_sequences):
    """Alignments of colabfold_search (one a3m per job in msas_colabfold)."""
    a3m_files = [msa_dir] if os.path.isfile(msa_dir) else sorted(
        os.path.join(msa_dir, name) for name in os.listdir(msa_dir) if name.endswith(".a3m"))
    msas = {}
    for a3m_file in a3m_files:
        with open(a3m_file, "r") as f:
            chains = split_colabfold_a3m(f.read())
        # the a3m of the job is the one aligning all the requested sequences
        if not set(sequences) <= set(chains):
            continue
        for sequence in sequences:
            paired_text, unpaired_text = chains[sequence]
            paired = a3m_text_to_msa(paired_text, max_sequences)
            unpaired_msa = a3m_text_to_msa(unpaired_text, max_sequences)
            msas[sequence] = stack([
                unpaired(unpaired_msa) if unpaired_msa is not None else None,
                # the query row of the paired block is dropped by the stacking
                row_keyed(paired) if paired is not None else None,
            ])
        break
    return msas

def read_msas(source, msa_dir, sequences, max_sequences):
    """Return {protein sequence: MSA} for the sequences found in the alignments of the source."""
    if source == "none" or not sequences:
        return {}
    if not os.path.exists(msa_dir):
        raise FileNotFoundError(f"MSA source '{source}' given but --msa_dir '{msa_dir}' does not exist")
    readers = {
        "afmassive": read_afmassive_msas,
        "alphafold3": read_alphafold3_msas,
        "colabfold": read_colabfold_msas,
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

def write_sample(result, seed, sample, output_dir, chains):
    name = f"esmf2_seed_{seed}_sample_{sample}"

    with open(os.path.join(output_dir, f"{name}.cif"), "w") as f:
        f.write(result.complex.to_mmcif())

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
