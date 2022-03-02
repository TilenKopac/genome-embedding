import json
import os.path
import pickle
import gzip

from Bio import Entrez, SeqIO
from Bio.Seq import UnknownSeq
from tqdm import tqdm

search_terms = [
    "Viruses[Organism] AND refseq[filter]",
    "Bacteria[Organism] AND refseq[filter]"
]
# sampling factors determine how many records will be left out
data_dirs = ["../data/viruses/fasta", "../data/bacteria/fasta"]
sampling_factors = [1, 4]

# set Entrez email and API key
try:
    with open("../entrez_credentials.json", "r") as file:
        login = json.load(file)
        if not login.get("email"):
            raise Exception("Field \"email\" in file \"entrez_credentials.json\" should not be empty")
        if not login.get("api_key"):
            print("Field \"api_key\" in file \"entrez_credentials.json\" is empty. Continuing without API key")
        Entrez.email = login.get("email")
        Entrez.api_key = login.get("api_key")
except FileNotFoundError:
    raise Exception("Please provide the \"entrez_credentials.json\" file with e-mail "
                    "and an optional API key to use in Entrez queries")

if not os.path.isdir("../data"):
    os.makedirs("../data")
for data_dir in data_dirs:
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)

for term, data_dir, sampling_factor in zip(search_terms, data_dirs, sampling_factors):
    print(f"Obtaining nucleotide records for search term '{term}'")

    # set esearch and efetch batch sizes
    esearch_batch = 1000 * sampling_factor
    efetch_batch = 100

    # get count of matching records
    handle = Entrez.esearch(db="nucleotide", term=term, retmax=0)
    results = Entrez.read(handle)
    handle.close()
    record_count = int(results["Count"])

    # progress bar
    pbar = tqdm(total=record_count // sampling_factor)

    # find matching record ids
    for i in range(0, record_count, esearch_batch):
        id_list = set()
        handle = Entrez.esearch(db="nucleotide", term=term, retstart=i, retmax=esearch_batch, idtype="acc",
                                usehistory="y", sort="accession")
        results = Entrez.read(handle)
        for ident in results["IdList"][::sampling_factor]:
            id_list.add(ident)

        # filter already obtained records
        existing_ids = set([filename.strip(".pkl.gz") for filename in os.listdir(data_dir)])
        to_fetch = id_list.difference(existing_ids)
        pbar.update(len(id_list.intersection(existing_ids)))

        if to_fetch:
            # fetch nucleotide DB records
            for j in range(0, len(to_fetch), efetch_batch):
                handle = Entrez.efetch(db="nucleotide", rettype="fasta", retstart=j, retmax=efetch_batch, id=to_fetch)
                for record in SeqIO.parse(handle, "fasta"):
                    if type(record.seq) != UnknownSeq:
                        with gzip.open(f"{data_dir}/{record.id}.pkl.gz", "wb") as file:
                            pickle.dump(record, file, protocol=4)
                    pbar.update(1)
                handle.close()

    pbar.close()
