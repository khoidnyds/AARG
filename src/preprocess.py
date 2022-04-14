import pandas as pd
from pyfaidx import Fasta
from collections import Counter
import json
from utils import run_subprocess, read_fasta_to_df
import numpy as np
import logging
from pathlib import Path
import networkx as nx
from Bio.SeqUtils.ProtParam import ProteinAnalysis


class ProcessCARD():
    """
    Generate 3 files:
    card_list.json: list of selected ARGs and their occurences
    card_map.json: mapping from accession to ARGs
    card.fasta: protein sequence of ARGs
    """

    def __init__(self, input, threshold, out_dir) -> None:
        self.path_card_seq = input.joinpath(
            "protein_fasta_protein_homolog_model.fasta")
        self.path_card_label = input.joinpath("aro_index.tsv")
        self.path_card_seq_clean = out_dir.joinpath("card.fasta")
        self.path_arg_list = out_dir.joinpath("card_list.txt")
        self.path_arg_map = out_dir.joinpath("card_map.tsv")
        self.threshold = threshold

    def process(self):
        # return self.path_card_seq_clean, self.path_arg_list, self.path_arg_map

        try:
            card_label = pd.read_csv(self.path_card_label, sep='\t')
            # some duplicate ARO in aro_index!
            card_label = card_label.drop_duplicates(
                subset='ARO Accession', keep="first")

            # take into account the top common ARGs
            drug_split = card_label['Drug Class'].str.replace(
                " antibiotic", "").str.strip()
            drug_split = ";".join(drug_split).split(";")
            drug_split = "-".join(drug_split).split("-")
            drug_split = Counter(drug_split)
            drug_split = {k: v for k, v in drug_split.items() if v >
                          self.threshold}
            drug_split = dict(
                sorted(drug_split.items(), key=lambda item: item[1], reverse=True))
            drug_labels = drug_split.keys()
            with open(self.path_arg_list, "w") as file:
                json.dump(drug_split, file)
        except Exception as e:
            logging.error("Can't generate card_list.json in preprocess.py")
            logging.error(e)

        # extract clean fasta
        try:
            arg_seq_accession = set()
            # >gb|ACT97415.1|ARO:3002999|CblA-1 [mixed culture bacterium AX_gF3SD01_15]
            with open(self.path_card_seq_clean, "w") as file:
                for seq in Fasta(str(self.path_card_seq)):
                    _, _, a, _ = seq.long_name.split("|")
                    if a in card_label['ARO Accession'].values:
                        file.write(f">{a}\n")
                        file.write(f"{seq}\n")
                        arg_seq_accession.add(a)

            # remove ARGs don't have sequence from mapping
            extra_accession = set(
                card_label['ARO Accession']).difference(arg_seq_accession)
            card_label = card_label[~card_label['ARO Accession'].isin(
                extra_accession)]
        except Exception as e:
            logging.error("Can't generate card.fasta in preprocess.py")
            logging.error(e)

        # extract mapping
        try:
            card_label[['ARO Accession', 'Drug Class']].to_csv(
                self.path_arg_map, index=False, sep="\t")
        except Exception as e:
            logging.error("Can't generate card_map.json in preprocess.py")
            logging.error(e)

        return self.path_card_seq_clean, self.path_arg_list, self.path_arg_map


class ProcessSTRING():
    """
    Generate35 files:
    alignment.tsv: STRING sequences align againts CARD 
    graph.gexf: store the graph for visualization 
        GRAPH - NODE: attributes (41) + label (list of categories)
              - EDGE: attribute (1)
              - ADJ: weight
    node_features.csv: features of nodes, nums of node * nums of features
    edge_features.csv: features of edges, nums of edge * nums of features
    adjacency.csv: adjacency matrix, nums of node * nums of node
    label.csv: labels of node, nums of node * nums of category
    """

    def __init__(self, path_string, path_card_seq, path_card_list, path_card_map, threshold, out_dir) -> None:
        self.path_card_seq = path_card_seq
        self.path_card_list = path_card_list
        self.path_card_map = path_card_map
        self.path_db = out_dir.joinpath("blastdb").joinpath("card")
        self.path_string_seq = path_string.joinpath(
            "protein.sequences.v11.5.fa")
        self.path_string_seq = Path("S_AUREUS/1280.protein.sequences.v11.5.fa")
        self.path_string_adj = path_string.joinpath(
            "protein.links.full.v11.5.txt")
        self.path_string_adj = Path(
            "S_AUREUS/1280.protein.links.full.v11.5.txt")
        self.path_alignment = out_dir.joinpath("alignment.tsv")
        self.path_alignment = Path("S_AUREUS/1280.protein.sequences.v11.5.tsv")
        self.path_nodes_list = out_dir.joinpath("nodes_list.csv")
        self.path_graph = out_dir.joinpath("graph.p")
        self.threshold = threshold

    def process(self):
        return self.path_graph

        # run alignment
        run_subprocess(
            f"makeblastdb -in {self.path_card_seq} -parse_seqids -blastdb_version 5 -dbtype prot -out {self.path_db}")
        run_subprocess(
            f'blastp -query {self.path_string_seq} -db {self.path_db} -num_threads 32 -mt_mode 1 -out {self.path_alignment} -outfmt "6 qseqid sseqid pident evalue bitscore"')

        # Opening JSON file
        card_map = pd.read_csv(self.path_card_map, sep='\t')
        card_map = card_map.set_index('ARO Accession')
        with open(self.path_card_list) as file:
            card_ls = json.load(file)
        aligned = pd.read_csv(self.path_alignment, sep='\t', names=[
                              'qseqid', 'sseqid', 'pident', 'evalue', 'bitscore'])
        aligned = aligned.merge(card_map, how='left',
                                left_on='sseqid', right_on='ARO Accession')
        aligned = aligned.dropna()

        highly_confident_ARG = aligned[aligned['pident'] > self.threshold]
        highly_confident_ARG = highly_confident_ARG.sort_values(
            'pident', ascending=False).drop_duplicates(['qseqid'])

        # create networkx
        G = nx.Graph()
        nodes_list = highly_confident_ARG['qseqid'].values
        G.add_nodes_from(nodes_list)
        string_adj = pd.read_csv(self.path_string_adj, sep=" ")
        string_adj = string_adj[string_adj['protein1'].isin(
            nodes_list) & string_adj['protein2'].isin(nodes_list)]
        G.add_weighted_edges_from([(e['protein1'], e['protein2'], float(
            e['combined_score'])) for _, e in string_adj.iterrows()])

        # remove isolated nodes from graph
        isolated_nodes = list(nx.isolates(G))
        logging.info(
            f"Remove isolated node from network: {len(isolated_nodes)}/{len(highly_confident_ARG)} = {len(isolated_nodes)/len(highly_confident_ARG)*100:.3f}%")
        G.remove_nodes_from(isolated_nodes)
        highly_confident_ARG = highly_confident_ARG[~highly_confident_ARG['qseqid'].isin(
            isolated_nodes)]

        # add node attributes
        node_features = {}
        node_label = {}
        string_seq = Fasta(str(self.path_string_seq))
        drug_labels = card_ls.keys()
        for n in G.nodes:
            # alignment
            temp = aligned[aligned['qseqid'] == n]
            drug_labels_score = [0]*len(drug_labels)
            for _, r in temp.iterrows():
                for i, d in enumerate(drug_labels):
                    if d in r['Drug Class']:
                        drug_labels_score[i] += r['bitscore']
            alignment_features = np.log(
                np.array(drug_labels_score)) / np.log(100)
            alignment_features = np.clip(alignment_features, 0.0, 1.0)
            # aa composition
            aa_composition = np.array(
                list(ProteinAnalysis(str(string_seq[n])).get_amino_acids_percent().values()))
            node_features[n] = np.concatenate(
                (alignment_features, aa_composition))

            n_labels = highly_confident_ARG[highly_confident_ARG['qseqid']
                                            == n]['Drug Class'].values
            concat_n_labels = ';'.join(n_labels)
            node_label[n] = [int(i in concat_n_labels) for i in drug_labels]
            ####################### node_label[n] = card_map.loc[n]['Drug']

        nx.set_node_attributes(G, node_features, name="features")
        nx.set_node_attributes(G, node_label, name="label")

        # add edge attributes
        # if edge has any of "neighborhood", "fusion", "cooccurence", edge attribute is 1 else 0
        string_adj = string_adj.set_index(['protein1', 'protein2'])
        edge_features = {}
        for e in G.edges:
            if string_adj.loc[e[0], e[1]]['neighborhood'] or string_adj.loc[e[0], e[1]]['fusion'] or string_adj.loc[e[0], e[1]]['cooccurence']:
                edge_features[e] = [1]
            else:
                edge_features[e] = [0]
        nx.set_edge_attributes(G, edge_features, name="features")

        # store graph
        info = nx.info(G).split("\n")
        logging.info(
            f"Graph: {info[2]}, {info[3]}, {info[4]}")
        nx.write_gpickle(G, self.path_graph)
        highly_confident_ARG = highly_confident_ARG.drop(
            ['Drug Class'], axis=1)
        highly_confident_ARG.to_csv(self.path_nodes_list, index=False)
        return self.path_graph
