# This script processes the ttl GND file into a graph structure.
import logging
import pickle

import networkx as nx
import pandas as pd
import requests
import rdflib
from rdflib.extras.external_graph_libs import rdflib_to_networkx_digraph
from tqdm import tqdm

# Set up logging.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
api_url = "https://lobid.org/gnd/{}.json"

def strip_uri(uri):
    return str(uri).split("/")[-1]

def get_gnd_data(gnd):
    response = requests.get(api_url.format(gnd))
    if response.status_code == 200:
        return response.json()

path_gnd = "data/GND-Subjects-plus.ttl"
out_path = "data/gnd-new.pickle"
metadata_path = "gnd_freq-used-meta-data.arrow"
columns = ["entitaetentyp", "idn", "gnd_identifier"]
gnd_meta = pd.read_feather(metadata_path, columns=columns)

meta_mapping = {
    idn: {"type": entitaetentyp, "gnd_id": strip_uri(gnd_identifier)} 
    for entitaetentyp, idn, gnd_identifier in gnd_meta.values
    }

# Read in graph in turtle format.
logger.info(f"Reading in graph from {path_gnd}")
g = rdflib.Graph()
g.parse(path_gnd)

# Convert to directed networkx graph.
logger.info("Converting to networkx graph")
dg = rdflib_to_networkx_digraph(g) 

# The URIs are not very informative, we will remove them.
alt = rdflib.term.URIRef('http://www.w3.org/2004/02/skos/core#altLabel')
pref = rdflib.term.URIRef("http://www.w3.org/2004/02/skos/core#prefLabel")
hidden = rdflib.term.URIRef("http://www.w3.org/2004/02/skos/core#hiddenLabel")

delete = []
relabel_mapping = {}
for (head, tail), value in tqdm(dg.edges.items(), desc="Processing graph.", leave=False):
    for triple in value["triples"]:
        relation_type = triple[1]

        if relation_type == alt:
            if "alt" not in dg.nodes[head]:
                dg.nodes[head]["alt"] = set()
            dg.nodes[head]["alt"].add(str(tail))
            delete.append(tail)
        if relation_type == pref:
            if "pref" not in dg.nodes[head]:
                dg.nodes[head]["pref"] = set()
            dg.nodes[head]["pref"].add(str(tail))
            delete.append(tail)
        if relation_type == hidden:
            if "hidden" not in dg.nodes[head]:
                dg.nodes[head]["hidden"] = set()
            dg.nodes[head]["hidden"].add(str(tail))
            delete.append(tail)

dg.remove_nodes_from(delete)

node_mapping = {}
for (head, tail, data) in tqdm(dg.edges(data=True), desc="Extracting node mapping.", leave=False):
    node_mapping[head] = strip_uri(head)
    node_mapping[tail] = strip_uri(tail)
    for head, rel, tail in data.pop("triples"):
        rel_type = rel.split("#")[-1]
    data["relation"] = rel_type
logger.info("Relabeling nodes.")
nx.relabel_nodes(dg, node_mapping, copy=False)

no_info_nodes = []
for node, data in tqdm(dg.nodes(data=True), desc="Adding meta information.", leave=False):
    if node in meta_mapping:
        meta_info = meta_mapping.get(node, {})
        data.update(meta_info)
    if not data:
        no_info_nodes.append(node)

number_noinfo = len(no_info_nodes)
if number_noinfo > 0:
    logger.warning(f"No information found for {number_noinfo} nodes. They will be removed.")
    dg.remove_nodes_from(no_info_nodes)
 
no_pref = []
logger.info(f"#Nodes: {dg.number_of_nodes()} #Edges: {dg.number_of_edges()}")

for node, data in tqdm(dg.nodes(data=True), desc="Checking for missing information.", leave=False):
    if "pref" not in data and "gnd_id" in data:
        no_pref.append(node)

logger.warning(f"Number of nodes without preferred name: {len(no_pref)}")

rec = 0
for node in tqdm(no_pref, desc="Recovering missing information.", leave=False):
    data = dg.nodes[node]
    gnd = data["gnd_id"]
    response = get_gnd_data(gnd)
    if response is not None:
        pref = response.get("preferredName")
        if pref is not None:
            data["pref"] = {pref} 
            rec += 1

print(f"Recovered {rec} of {len(no_pref)} missing preferred names.")

# Save the graph.
logger.info("Saving graph.")
pickle.dump(dg, open(out_path, 'wb'))