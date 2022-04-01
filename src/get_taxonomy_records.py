import gzip
import json
import os.path
import pickle

from Bio import Entrez, SeqIO
from tqdm import tqdm


def get_taxonomy(rec):
    tax = {}
    for element in rec["LineageEx"]:
        tax[str(element["Rank"])] = str(element["ScientificName"])
    return tax


nuc_records_dirs = ["../data/viruses/fasta", "../data/bacteria/fasta"]
tax_records_dirs = ["../data/viruses/taxonomy", "../data/bacteria/taxonomy"]
efetch_batch = 100

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
for data_dir in nuc_records_dirs:
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)
for data_dir in tax_records_dirs:
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)

for nuc_records_dir, tax_records_dir in zip(nuc_records_dirs, tax_records_dirs):
    print(f"Obtaining taxonomy records for organisms in directory '{nuc_records_dir}'")

    # get count of matching records
    record_count = len(os.listdir(nuc_records_dir))

    # progress bar
    pbar = tqdm(total=record_count)
    pbar.update(len(os.listdir(tax_records_dir)))

    # filter already obtained records
    to_fetch = [filename.strip(".pkl.gz") for filename in
                set(os.listdir(nuc_records_dir)).difference(set(os.listdir(tax_records_dir)))]

    # fetch GenBank and taxonomy records
    for i in range(0, record_count, efetch_batch):
        nuc_tax_id_pairs = dict()

        # GenBank records
        handle = Entrez.efetch(db="nucleotide", rettype="gb", retstart=i, retmax=efetch_batch, id=to_fetch)
        tax_ids = []
        for record in SeqIO.parse(handle, "gb"):
            source_orgs = list(filter(lambda feature: feature.type == "source", record.features))
            if source_orgs:
                try:
                    tax_id = source_orgs[0].qualifiers["db_xref"][0].strip("taxon:")
                    tax_ids.append(tax_id)
                    nuc_tax_id_pairs[tax_id] = record.id
                except KeyError:
                    continue
        handle.close()

        # taxonomy records
        handle = Entrez.efetch(db="taxonomy", retmode="xml", id=tax_ids)
        results = Entrez.read(handle)
        for record in results:
            try:
                taxonomy = get_taxonomy(record)
                nuc_id = nuc_tax_id_pairs[record["TaxId"]]
                with gzip.open(f"{tax_records_dir}/{nuc_id}.pkl.gz", "wb") as file:
                    pickle.dump(taxonomy, file, protocol=4)
            except KeyError:
                continue
        handle.close()

        pbar.update(efetch_batch)
    pbar.close()
