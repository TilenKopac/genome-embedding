import os
import pickle
import random
from enum import Enum

from commons import nucleotide_encoding, input_output


class TaxonomyDataset:

    def __init__(self, data_dir, tax_rank, encoder, sampler):
        super(TaxonomyDataset, self).__init__()
        self.tax_recs_dir = os.path.join(data_dir, "taxonomy")
        self.nuc_recs_dir = os.path.join(data_dir, "fasta")
        try:
            with open(os.path.join(data_dir, "taxa_index.pkl"), "rb") as file:
                self.taxa_index = pickle.load(file)

        except FileNotFoundError:
            raise Exception(f"Taxa index not found on path {os.path.join(data_dir, 'taxa_index')}")

        self.tax_rank = tax_rank
        self.encoder = encoder
        self.sampler = sampler

    def get_record(self):
        filenames = os.listdir(self.tax_recs_dir)
        random.shuffle(filenames)
        for filename in filenames:
            tax_rec = input_output.read_from_disk(self.tax_recs_dir, filename)
            nuc_rec = input_output.read_from_disk(self.nuc_recs_dir, filename)
            if not tax_rec.get(self.tax_rank.value) or not nucleotide_encoding.sequence_valid(nuc_rec.seq):
                continue
            seq_encoded = nucleotide_encoding.integer_encode(nuc_rec.seq)
            tax_encoded = self.taxa_index[self.tax_rank.value][tax_rec[self.tax_rank.value]]

            yield seq_encoded, tax_encoded


class TaxonomicRankEnum(Enum):
    GENUS = "genus"
    SPECIES = "species"


dataset = TaxonomyDataset(
    "../data/viruses",
    TaxonomicRankEnum.GENUS,
    None,
    None
)
seq, tax_rank = next(dataset.get_record())
print(tax_rank)
