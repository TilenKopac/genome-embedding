import os

from Bio.Seq import Seq
from tqdm import tqdm

from definitions import DATA_DIR

dataset_dir = os.path.join(DATA_DIR, "bacteria_661k_assemblies_balanced", "fasta")
for subset in ["train", "test", "val"]:
    with open(os.path.join(dataset_dir, f"{subset}.csv"), "rt") as in_file:
        with open(os.path.join(dataset_dir, f"{subset}_with_reverse_complements.csv"), "wt") as out_file:
            for line in tqdm(in_file, desc=f"Reverse complementing {subset} subset"):
                org_id, seq_str = line.strip().split(",")
                seq = Seq(seq_str)
                out_file.write(f"{org_id},{seq}\n")
                out_file.write(f"{org_id},{seq.reverse_complement()}\n")
