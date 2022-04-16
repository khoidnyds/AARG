#!/bin/sh
#SBATCH --time=100:00:00
#SBATCH --partition=normal_q
#SBATCH -n 32
#SBATCH --mem=100G
#SBATCH --account=aipmm

export CARD_RAW="CARD/protein_fasta_protein_homolog_model.fasta"
export CARD_CLEAN="results/04-13--19-08-08/card.fasta"
export CARD_DB="results/04-13--19-08-08/blastdb/card"
export ALIGNMENT_RAW="results/04-13--19-08-08/alignments_raw.tsv"
export ALIGNMENT_HIGH_IDENTITY="results/04-13--19-08-08/alignments_high_identity.tsv"
export NODES_NON_DUP="results/04-13--19-08-08/nodes_non_dup.txt"
export STRING_SEQ="STRING/protein.sequences.v11.5.fa"
export STRING_LINK_FULL="STRING/protein.links.full.v11.5.txt"
export STRING_LINK_FULL="S_AUREUS/1280.protein.links.full.v11.5.txt"
export EDGES="results/04-13--19-08-08/edges.txt"
export NODES="results/04-13--19-08-08/nodes.txt"
export ALIGNMENT="results/04-13--19-08-08/alignments.tsv"

export THREAD="32"
export IDENTITY="90"

# clean header fasta file
awk -F'|' '/^>/ { print ">"$3; next } 1' $CARD_RAW > $CARD_CLEAN

# # run alignment
makeblastdb -in $CARD_CLEAN -out $CARD_DB -parse_seqids -blastdb_version 5 -dbtype prot
blastp -query $STRING_SEQ -db $CARD_DB -out $ALIGNMENT_RAW -num_threads $THREAD -mt_mode 1 -outfmt "6 qseqid sseqid pident evalue bitscore"

# extract high identity alignment
awk -v iden=$IDENTITY '{if($3>iden) print $0}' $ALIGNMENT_RAW > $ALIGNMENT_HIGH_IDENTITY

# drop duplicated row
awk '!seen[$1]++ {print $1}' $ALIGNMENT_HIGH_IDENTITY > $NODES_NON_DUP

# filter out edges having nodes in nodes.txt
awk 'NR==FNR{a[$1];next}($1 in a)&&($2 in a)' $NODES_NON_DUP $STRING_LINK_FULL > $EDGES

# filter out nodes don't have connection
awk 'NR==FNR{a[$1]+a[$2];next}($1 in a)' $STRING_LINK_FULL $NODES_NON_DUP > $NODES

# filter out alignment
awk 'NR==FNR{a[$1];next}($1 in a)' $NODE $ALIGNMENT_RAW > $ALIGNMENT
