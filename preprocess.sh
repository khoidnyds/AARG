#!/bin/sh
#SBATCH --time=100:00:00
#SBATCH --partition=normal_q
#SBATCH -n 32
#SBATCH --mem=100G
#SBATCH --account=aipmm

export DATE="04-13--19-08-08"
export THREAD="32"
export IDENTITY="90"

export CARD_RAW="CARD/protein_fasta_protein_homolog_model.fasta"
export CARD_CLEAN="results/$DATE/card.fasta"
export CARD_DB="results/$DATE/blastdb/card"

export ALIGNMENTS_RAW="results/$DATE/string_alignments_raw.tsv"
export ALIGNMENTS_HIGH_IDENTITY="results/$DATE/string_alignments_high_identity.tsv"
export ALIGNMENTS="results/$DATE/string_alignments.tsv"

export NODES_NON_DUP="results/$DATE/string_nodes_no_dup.txt"
export STRING_SEQ="STRING/protein.sequences.v11.5.fa"
export STRING_LINK_FULL="STRING/protein.links.full.v11.5.txt"
export EDGES="results/$DATE/string_edges.txt"
export NODES="results/$DATE/string_nodes.txt"

echo "Result dir: results/$DATE, set thread=$THREAD, set alignment identity=$IDENTITY"


# echo "clean header fasta file"
# awk -F'|' '/^>/ { print ">"$3; next } 1' $CARD_RAW > $CARD_CLEAN

# echo "run alignment"
# makeblastdb -in $CARD_CLEAN -out $CARD_DB -parse_seqids -blastdb_version 5 -dbtype prot
# blastp -query $STRING_SEQ -db $CARD_DB -out $ALIGNMENTS_RAW -num_threads $THREAD -mt_mode 1 -outfmt "6 qseqid sseqid pident evalue bitscore"

echo "Extract high identity alignment"
awk -v iden=$IDENTITY '{if($3>iden) print $0}' $ALIGNMENTS_RAW > $ALIGNMENTS_HIGH_IDENTITY

echo "Drop duplicated nodes"
awk '!seen[$1]++ {print $1}' $ALIGNMENTS_HIGH_IDENTITY > $NODES_NON_DUP

echo "Filter out edges having nodes in nodes.txt"
awk 'NR==FNR{a[$1];next}($1 in a)&&($2 in a)' $NODES_NON_DUP $STRING_LINK_FULL > $EDGES

echo "Filter out nodes don't have connection"
awk 'NR==FNR{a[$1]+a[$2];next}($1 in a)' $STRING_LINK_FULL $NODES_NON_DUP > $NODES

echo "Filter out alignments for network only"
awk 'NR==FNR{a[$1];next}($1 in a)' $NODES $ALIGNMENTS_RAW > $ALIGNMENTS

echo "Finish!"
