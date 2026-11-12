import os
import re
import csv
import argparse
import random
import string
import logging
import fcntl
import numpy as np
import pickle
import glob
import sys
import pandas as pd

from esm.models.esmfold2 import (
    ESMFold2InputBuilder,
    ProteinInput,
    DNAInput,
    RNAInput,
    LigandInput,
    StructurePredictionInput,
    MSA,
    EsmFold2Model,
)

# ==========================================
# 0. HELPER FUNCTIONS
# ==========================================
def parse_fasta(filepath):
    """Reads a FASTA file and returns a list of sequence strings."""
    sequences = []
    current_seq = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_seq:
                    sequences.append("".join(current_seq))
                    current_seq = []
            elif line:
                current_seq.append(line)
        if current_seq:
            sequences.append("".join(current_seq))
    return sequences

# ==========================================
# 1. ARGUMENT PARSING & SETUP
# ==========================================
parser = argparse.ArgumentParser(description="Run Sequential ESMFold2 Array Task")
parser.add_argument("--fasta_file", type=str, required=True, help="Path to the input FASTA file")
parser.add_argument("--run_name", type=str, required=True, help="Subfolder name for this specific execution")
parser.add_argument("--task_id", type=int, required=True, help="Explicit Slurm Array Task ID")
parser.add_argument("--n_seeds", type=int, default=1, help="Number of seeds used for structure predictions")
parser.add_argument("--n_loops", type=int, default=10, help="Number of loops used for structure predictions")
parser.add_argument("--n_samples", type=int, default=5, help="Number of diffusion samples used for structure predictions")
parser.add_argument("--n_sampling_steps", type=int, default=100, help="Number of sampling steps used for structure predictions")
parser.add_argument("--msas_from_tool", type=str, choices=["AlphaFold2", "noMsas"], default="noMsas", help="Which MSA tool outputs to use. Select 'noMsas' to fold without alignments.")
parser.add_argument("--molecule_types", type=str, required=True, help="Comma-separated list of types for each sequence: protein, dna, rna, or ligand (e.g., 'protein,protein,dna')")

# ==========================================
# 2. SEQUENTIAL DIRECTORY RESOLUTION
# ==========================================
def prepare_output(run_name, base_out_dir, base_log_dir):
    actual_run_name = run_name
    out_dir = os.path.join(base_out_dir, actual_run_name)
    log_dir = os.path.join(base_log_dir, actual_run_name)
    
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    return out_dir, log_dir, actual_run_name

def prepare_input(fasta_file, task_id, base_out_dir, msas_from_tool, molecule_types_str):
    # ==========================================
    # 3. DYNAMIC SEQUENCE & MSA LOADING
    # ==========================================
    logging.info(f"[Task {task_id}] Parsing sequences from {fasta_file}...")
    sequences = parse_fasta(fasta_file)
    chain_alphabet = string.ascii_uppercase
    
    mol_types = [t.strip().lower() for t in molecule_types_str.split(",")]

    type_to_class = {
        "protein": ProteinInput,
        "dna": DNAInput,
        "rna": RNAInput,
        "ligand": LigandInput
    }

    chain_inputs = []
    unique_protein_sequences = []

    msa_base_dir = os.path.join(base_out_dir, "msas")
    chain_folder_map = {}
    # unique_idx: {"id": [chain_id], "folder": folder_id}

    for idx, (seq, mol_type) in enumerate(zip(sequences, mol_types)):
        chain_id = chain_alphabet[idx] 

        if mol_type not in type_to_class:
            logging.warning(f"[Task {task_id}] Unknown molecule type '{mol_type}'. Defaulting to 'protein'.")
            mol_type = "protein"

        InputClass = type_to_class[mol_type]

        if mol_type == "protein":
            if seq not in unique_protein_sequences:
                unique_protein_sequences.append(seq)

            unique_idx = unique_protein_sequences.index(seq)
            if unique_idx not in chain_folder_map:
                chain_folder_map[unique_idx] = {"id": [chain_id], "folder": chain_id}
            else:
                chain_folder_map[unique_idx]["id"].append(chain_id)
            folder_id = chain_folder_map[unique_idx]["folder"]

            #folder_id = chain_alphabet[unique_idx] 

            if msas_from_tool == "AlphaFold2":
                msa_path_bfd = os.path.join(msa_base_dir, folder_id, "bfd_uniref_hits.a3m")
                msa_path_mgnify = os.path.join(msa_base_dir, folder_id, "mgnify_hits.sto")
                msa_path_uniref90 = os.path.join(msa_base_dir, folder_id, "uniref90_hits.sto")

                if (os.path.exists(msa_path_bfd)) and (os.path.exists(msa_path_mgnify)) and (os.path.exists(msa_path_uniref90)):
                    logging.info(f"[Task {task_id}] Loading AlphaFold2 MSA for Protein Chain {chain_id} from mapped folder {folder_id}...")
                    
                    msa_bfd = MSA.from_a3m(msa_path_bfd, remove_insertions=True, max_sequences=200000)
                    msa_mgnify = MSA.from_stockholm(msa_path_mgnify, remove_insertions=True, max_sequences=200000)
                    msa_uniref90 = MSA.from_stockholm(msa_path_uniref90, remove_insertions=True, max_sequences=200000)

                    logging.info(f"[Task {task_id}] Combining the alignments...")
                    msa = MSA.stack([msa_bfd, msa_mgnify, msa_uniref90])

                    chain_inputs.append(ProteinInput(id=chain_id, sequence=msa.query, msa=msa))
                else:
                    logging.info(f"[Task {task_id}] WARNING: AlphaFold2 requested but MSAs missing in mapped folder {folder_id}. Folding Protein Chain {chain_id} as single sequence.")
                    chain_inputs.append(ProteinInput(id=chain_id, sequence=seq))
            else:
                logging.info(f"[Task {task_id}] Folding Protein Chain {chain_id} as single sequence without alignments.")
                chain_inputs.append(ProteinInput(id=chain_id, sequence=seq))

        elif mol_type == "ligand": 
            logging.info(f"[Task {task_id}] Processing {mol_type.upper()} Chain {chain_id} (No MSAs used).")
            chain_inputs.append(LigandInput(id=chain_id, smiles=seq))

        else:
            logging.info(f"[Task {task_id}] Processing {mol_type.upper()} Chain {chain_id} (No MSAs used).")
            chain_inputs.append(InputClass(id=chain_id, sequence=seq))

    spi = StructurePredictionInput(sequences=chain_inputs)
    return spi

def set_occupancy(mmcif_string, occupancy="1.00"):
    lines = mmcif_string.splitlines()
    output_lines = []
    i = 0

    while i < len(lines):
        line = lines[i]

        if line.strip() != "loop_":
            output_lines.append(line)
            i += 1
            continue

        loop_start = i
        tags = []
        tag_lines = []

        output_lines.append(line)
        i += 1

        while i < len(lines) and lines[i].strip().startswith("_"):
            tag = lines[i].strip()
            tags.append(tag)
            tag_lines.append(lines[i])
            i += 1

        is_atom_site_loop = any(tag.startswith("_atom_site.") for tag in tags)

        if not is_atom_site_loop:
            output_lines.extend(tag_lines)
            continue

        occupancy_tag = "_atom_site.occupancy"

        if occupancy_tag in tags:
            occupancy_idx = tags.index(occupancy_tag)
            output_lines.extend(tag_lines)
        else:
            occupancy_idx = len(tags)
            output_lines.extend(tag_lines)
            output_lines.append(occupancy_tag)
            tags.append(occupancy_tag)

        while i < len(lines):
            stripped = lines[i].strip()

            if (
                stripped == ""
                or stripped == "#"
                or stripped == "loop_"
                or stripped.startswith("data_")
                or stripped.startswith("save_")
                or stripped.startswith("_")
            ):
                break

            parts = lines[i].split()

            if occupancy_tag in tags and len(parts) > occupancy_idx:
                if len(parts) == len(tags):
                    parts[occupancy_idx] = occupancy
                elif len(parts) == len(tags) - 1:
                    parts.append(occupancy)

            output_lines.append(" ".join(parts))
            i += 1

        continue

    output_lines.extend(lines[i:])

    return "\n".join(output_lines) + "\n"

def output_single_seed_samples(result, current_seed, sample_id, task_id, out_dir):
    plddt_val = float(result.plddt.mean())
    ptm_val = float(result.ptm)
    iptm_val = float(result.iptm)

    if hasattr(result, "pair_chains_iptm") and result.pair_chains_iptm is not None:
        pair_matrix = result.pair_chains_iptm.tolist()
        pair_chains_str = str([[round(v, 4) for v in row] for row in pair_matrix])
    else:
        pair_chains_str = "N/A"

    logging.info(f"[Task {task_id}] pLDDT: {plddt_val}, pTM: {ptm_val}, ipTM: {iptm_val}")

    # ==========================================
    # 5. OUTPUT CIF, PICKLE AND CSV METRICS
    # ==========================================
    row = {
        "task_id": task_id,
        "seed": current_seed,
        "sample": sample_id,
        "pLDDT": plddt_val,
        "pTM": ptm_val,
        "ipTM": iptm_val,
        "pair_chains_ipTM": pair_chains_str
    }

    logging.info(f"[Task {task_id}] Successfully wrote metrics to CSV.")

    full_model_name = f"esmf2_seed_{current_seed}_sample_{sample_id}_pred_{task_id}"
    pickle_file = os.path.join(out_dir, f"{full_model_name}.pkl")

    pickle_content = {}
    pickle_content["predicted_aligned_error"] = result.pae.float().numpy()
    pickle_content["plddt"] = result.plddt.float().numpy()
    pickle_content["max_predicted_aligned_error"] = np.max(pickle_content["predicted_aligned_error"])
    with open(pickle_file, 'wb') as outputpickle:
        pickle.dump(pickle_content, outputpickle)
    logging.info(f"[Task {task_id}] Pickle production at {pickle_file}")

    out_filepath = os.path.join(out_dir, f"{full_model_name}.cif")
    mmcif_string = result.complex.to_mmcif()
    mmcif_string = set_occupancy(mmcif_string)

    with open(out_filepath, "w") as f:
        f.write(mmcif_string)

    logging.info(f"[Task {task_id}] Saved structure to {out_filepath}")
    return row

def predict_single_seed(spi, model, current_seed, task_id, out_dir, n_loops, n_samples, n_sampling_steps):
    # ==========================================
    # 4. RUNNING INFERENCE
    # ==========================================
    samples_preds = []
    logging.info(f"[Task {task_id}] Initiating prediction (Seed: {current_seed})...")
    fold_output = ESMFold2InputBuilder().fold(
        model,
        spi,
        num_loops=n_loops,
        num_sampling_steps=n_sampling_steps,
        num_diffusion_samples=n_samples,
        seed=current_seed,
    )
    fold_results = []
    if isinstance(fold_output, list):
        fold_results.extend(fold_output)
    else:
        fold_results.append(fold_output)

    seed_rows = []
    for sample_id, result in enumerate(fold_results):
        summary_row = output_single_seed_samples(
            result, current_seed, sample_id, task_id, out_dir
        )
        seed_rows.append(summary_row)

    return seed_rows


if __name__ == "__main__":
    args = parser.parse_args()
    
    # --- HARD STOP CHECK ---
    # Parse the molecule types string immediately
    mol_types = [t.strip().lower() for t in args.molecule_types.split(",")]
    
    # Check if AlphaFold2 MSAs are requested alongside non-protein molecules
    if args.msas_from_tool == "AlphaFold2" and any(t != "protein" for t in mol_types):
        print("AlphaFold2 alignments impossible to handle when not only proteins")
        sys.exit(1) # Exit cleanly with an error status before anything else runs
    # -----------------------
    
    task_id = args.task_id
    run_name = args.run_name
    fasta_file = args.fasta_file
    msas_from_tool = args.msas_from_tool
    n_seeds = args.n_seeds
    n_loops = args.n_loops
    n_samples = args.n_samples
    n_sampling_steps = args.n_sampling_steps
    molecule_types_str = args.molecule_types

    fasta_basename = os.path.splitext(os.path.basename(args.fasta_file))[0]
    base_out_dir = os.path.join(".", "output", fasta_basename)
    base_log_dir = os.path.join(".", "log", fasta_basename)

    os.makedirs(base_out_dir, exist_ok=True)
    os.makedirs(base_log_dir, exist_ok=True)
    out_dir, log_dir, actual_run_name = prepare_output(
        run_name,
        base_out_dir,
        base_log_dir
    )
    log_filepath = os.path.join(log_dir, f"task_{task_id}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(log_filepath),
            logging.StreamHandler()
        ]
    )

    logging.info(f"[Task {task_id}] Output directory resolved to: {out_dir}")
    logging.info(f"[Task {task_id}] Loading ESMFold2 model onto the GPU...")
    nn_model = EsmFold2Model.from_pretrained("biohub/ESMFold2", device="cuda").eval()
    logging.info(f"[Task {task_id}] Model loaded successfully.")

    spi = prepare_input(fasta_file, task_id, base_out_dir, msas_from_tool, molecule_types_str)

    # ==========================================
    # MSA VERIFICATION CHECK
    # ==========================================
    logging.info(f"[Task {task_id}] --- VERIFYING MSA INPUTS ---")
    for chain in spi.sequences:
        if hasattr(chain, "msa") and chain.msa is not None:
            msa_depth = len(chain.msa.headers)
            logging.info(f"-> Chain {chain.id} ({type(chain).__name__}): MSA loaded successfully with {msa_depth} sequences.")
        else:
            logging.info(f"-> Chain {chain.id} ({type(chain).__name__}): NO MSA FOUND / APPLICABLE. Folding as single sequence.")
    logging.info(f"[Task {task_id}] ----------------------------")

    df_rows = []
#    rng = random.Random(42)
    used_seeds = random.sample(range(1, 1_000_000), n_seeds)
    for seed_idx, used_seed in enumerate(used_seeds):
        #used_seed = random.randint(1, 1000000)
        logging.info(
            f"[Task {task_id}] Starting seed {seed_idx + 1}/"
            f"{n_seeds}: {used_seed}"
        )
        seed_rows = predict_single_seed(
            spi,
            nn_model,
            used_seed,
            task_id,
            out_dir,
            n_loops,
            n_samples,
            n_sampling_steps
        )
        df_rows.extend(seed_rows)

    metrics = pd.DataFrame(df_rows)
    print(metrics)
    csv_filename = os.path.join(out_dir, f"ranking_esmf2_{task_id}.csv")
    logging.info(f"[Task {task_id}] Writing metrics to {csv_filename}...")
    metrics.to_csv(csv_filename, index=False)

