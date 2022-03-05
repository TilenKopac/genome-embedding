import gzip
import os
import pickle
from collections import defaultdict

from tqdm import tqdm

tax_recs_dir = "../data/viruses/taxonomy"
index_dir = "../data/viruses"
index_filename = "taxa_index.pkl"
if os.path.isfile(f"{index_dir}/{index_filename}"):
    with open(f"{index_dir}/{index_filename}", "rb") as file:
        taxa_index = pickle.load(file)
        counters = defaultdict(int)
        for rank, taxa in taxa_index.items():
            counters[rank] = list(taxa.values())[-1] + 1
else:
    taxa_index = defaultdict(dict)
    counters = defaultdict(int)

for filename in tqdm((os.listdir(tax_recs_dir))):
    with gzip.open(f"{tax_recs_dir}/{filename}", "rb") as file:
        taxonomy = pickle.load(file)
        for rank, taxon in taxonomy.items():
            if taxon not in taxa_index[rank]:
                taxa_index[rank][taxon] = counters[rank]
                counters[rank] += 1

with open(f"{index_dir}/{index_filename}", "wb") as file:
    pickle.dump(taxa_index, file, protocol=4)
