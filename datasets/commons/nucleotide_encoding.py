nuc_integer_enc = {
    "A": 0,
    "G": 1,
    "C": 2,
    "T": 3,
    "N": 4}
valid_chars = nucleotides = set(nuc_integer_enc.keys())
nucleotides.remove("N")


def sequence_valid(sequence):
    if set(sequence).difference(valid_chars):
        # record contains invalid nucleotide characters
        return False
    return True


def integer_encode(sequence):
    return [nuc_integer_enc[nuc] for nuc in sequence]
