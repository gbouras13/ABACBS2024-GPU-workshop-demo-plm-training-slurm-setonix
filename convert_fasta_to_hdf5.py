
import argparse
from loguru import logger
import h5py
from Bio import SeqIO


def convert_fasta_to_hdf5(fasta_file, hdf5_file):
    sequences = list(SeqIO.parse(fasta_file, "fasta"))
    logger.info(f"Beginning converting {len(sequences)} sequences")
    with h5py.File(hdf5_file, 'w') as f:
        f.create_dataset('sequences', (len(sequences),), dtype=h5py.special_dtype(vlen=str))
        for i, record in enumerate(sequences):
            clean_sequence = str(record.seq).replace('*', '')
            f['sequences'][i] = clean_sequence
    logger.info(f"Converted {len(sequences)} sequences to {hdf5_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f','--fasta',
                        action="store",
                        help='training dataset FASTA')
    parser.add_argument('--hdf5_file',
                        action="store",
                        help='output hdf5 file')
    args = parser.parse_args()

    convert_fasta_to_hdf5(args.fasta, args.hdf5_file)

if __name__ == '__main__':
    main()