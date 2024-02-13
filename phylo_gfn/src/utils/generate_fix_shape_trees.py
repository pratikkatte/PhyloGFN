"""
Generate fixed shape trees from UFBOOST results
"""

import numpy as np
from Bio import SeqIO
from collections import Counter
import networkx as nx
from ete3 import Tree


fasta_file = 'vbpi-torch/unrooted/data/ufboot_data_DS1-11/DS1/DS1.fasta'
ufboot_file = 'vbpi-torch/unrooted/data/ufboot_data_DS1-11/DS1/DS1_ufboot_rep_{}'
# number of minimum presence required
good_lines_threshold = 2

# parse taxa
records = SeqIO.parse(fasta_file, 'fasta')
taxas = [x.id for x in records]
n_leaves = len(taxas)
n = n_leaves

# get all records
lines = []
for idx in range(1, 11):
    with open(ufboot_file.format(idx), 'r') as input_file:
        lines += input_file.readlines()
taxa_dict = {key: idx for idx, key in enumerate(taxas)}
lines_counts = Counter((lines))

# threshold by num of presence
good_lines = [key for key in lines_counts if lines_counts[key] >= good_lines_threshold]

ufboost_trees = []
for l in list(set(good_lines)):
    tree = Tree(l[:-1])
    for idx, node in enumerate(tree.traverse("preorder")):
        if node.name != '':
            node.name = taxa_dict[node.name.replace("'", "")]
        else:
            node.name = n
            n += 1

    n_nodes = 2 * n_leaves - 2

    nx_tree = nx.Graph()
    node_names = [n for n in range(n_nodes)]
    leaves = node_names[:n_leaves]

    # Add nodes to the graph
    for node in leaves:
        nx_tree.add_node(node, type='leaf')
    for node in node_names[n_leaves:-1]:
        nx_tree.add_node(node, type='internal')
    nx_tree.add_node(2 * n_leaves - 3, type='root')

    for idx, node in enumerate(tree.traverse("preorder")):
        if not node.is_leaf():
            children = []
            for child_node in node.children:
                nx_tree.add_edge(node.name, child_node.name, t=0.002)
                nx_tree.add_node(child_node.name, parent=node.name)

                children.append(child_node.name)
            nx_tree.add_node(node.name, children=children)

    ufboost_trees.append(nx_tree)
