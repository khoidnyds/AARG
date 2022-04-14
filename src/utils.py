from subprocess import Popen, PIPE
from pyfaidx import Fasta
import pandas as pd


def run_subprocess(cmd: str, get_stdout=False):
    if get_stdout:
        p = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)
        out, err = p.communicate()
        out = out.decode().strip()
        err = err.decode().strip()
        if out != "":
            return out
        elif err != "":
            return err
        else:
            return ""
    else:
        p = Popen(cmd, shell=True)
        p.wait()


def read_fasta_to_df(file):
    accessions, sequences = [], []
    for seq in Fasta(str(file)):
        accessions.append(seq.name)
        sequences.append(str(seq))
    return pd.DataFrame({"ARO Accession": accessions,
                         "Sequence": sequences}).set_index('ARO Accession')
