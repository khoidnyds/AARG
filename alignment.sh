#!/bin/sh
#SBATCH --time=100:00:00
#SBATCH --partition=normal_q
#SBATCH -n 32
#SBATCH --mem=100G
#SBATCH --account=aipmm

# python src/main.py

# makeblastdb -in results/04-13--19-08-08/card.fasta -out results/04-13--19-08-08/blastdb/card -parse_seqids -blastdb_version 5 -dbtype prot
blastp -query STRING/protein.sequences.v11.5.fa -db results/04-13--19-08-08/blastdb/card -out STRING/string_alignments_blast_raw.tsv -num_threads 32 -mt_mode 1 -outfmt "6 qseqid sseqid pident evalue bitscore"

# diamond makedb --in results/04-13--19-08-08/card.fasta -d results/04-13--19-08-08/diamond/card.dmnd
# diamond blastp --db results/04-13--19-08-08/diamond/card.dmnd --query STRING/protein.sequences.v11.5.fa --verbose --sensitive -o STRING/string_alignments_diamond_raw.tsv
