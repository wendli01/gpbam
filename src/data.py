from typing import Sequence, Optional, Tuple, Dict

import os
import subprocess

import numpy as np
import pdfplumber
import networkx as nx
from neo4j import GraphDatabase
import pandas as pd
import pickle
import dgl

from lxml import etree
from tqdm.auto import tqdm


def load_pdfs(dir: str = 'data/01_Sachverhalte', ext: str = 'pdf', remove_header: bool = True) -> Sequence[
    Sequence[str]]:
    fns = sorted([os.path.join(dir, f) for f in os.listdir(dir) if f.endswith('.' + ext)])

    all_docs = []
    for fn in tqdm(fns, total=len(fns), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
        with pdfplumber.open(fn) as pdf:
            pages = [[l['text'] for l in page.extract_text_lines(page)] for page in pdf.pages]

        if remove_header:
            pages = _remove_headers(pages)

        all_docs.append(pages)

    return ['\n'.join(['\n'.join(lines) for lines in pages]) for pages in all_docs]


def _remove_headers(pages: Sequence[Sequence[str]], max_len: int = 10) -> Sequence[Sequence[str]]:
    if len(pages) == 1:
        return pages

    min_lines = np.min([len(p) for p in pages])

    header_found = False
    for header_len in range(min(max_len, min_lines) - 1, 0, -1):
        candidates = [p[header_len] for p in pages]
        if len(candidates[0]) >= 5 and all([cand == candidates[0] for cand in candidates[1:]]):
            header_found = True
            break
        else:
            continue
    if not header_found:
        return pages

    return [page[header_len + 1:] for page in pages]


def from_neo4j(host: str = 'bolt://localhost:7687', auth=("neo4j", "neo4j"),
               label_mapping: dict = (('Legislation', 'Law'),)) -> nx.MultiDiGraph:
    with GraphDatabase.driver(host, auth=auth).session() as session:
        node_records = session.run("MATCH (n) RETURN n")
        nodes = list(node_records.graph()._nodes.values())

    G = nx.MultiDiGraph()

    label_mapping = {on: nn for on, nn in label_mapping}

    for node in nodes:
        properties = {**{k: str(v) for k, v in node._properties.items()}}
        label = None if len(node._labels) < 1 else list(node._labels)[0]
        properties['labels'] = label if label not in label_mapping else label_mapping[label]
        G.add_node(node.id, **properties)

    with GraphDatabase.driver(host, auth=auth).session() as session:
        rel_records = session.run("MATCH ()-[r]->() RETURN r")
        rels = list(rel_records.graph()._relationships.values())

    for rel in rels:
        G.add_edge(rel.start_node.id, rel.end_node.id, key=rel.id, type=rel.type, properties=rel._properties)

    return G


def load_old_graph(nx_dump_fp: Optional[str] = 'data/old/old_ref.pkl', max_expression_ratio: float = .5) -> Tuple[
    dgl.DGLHeteroGraph, pd.DataFrame, pd.DataFrame]:
    def construct_data_df(ntype: str):
        return pd.DataFrame([G.nodes[n] for n in type_nodes[ntype]]).drop(columns=['labels'])

    with open(nx_dump_fp, 'rb') as fb:
        G = pickle.load(fb)

    node_types = nx.get_node_attributes(G, 'labels')
    node_types = {n: node_types[n] for n in sorted(node_types)}

    node_counts = pd.Series(list(node_types.values())).value_counts()
    ntypes = set(node_counts.index)
    assert ntypes == {'Case', 'Court', 'Law'}, 'Graph needs Case, Court and Law nodes'

    type_nodes = {nt: [n for n, nt_ in node_types.items() if nt == nt_] for nt in ntypes}
    num_nodes_dict = {nt: node_counts[nt] for nt in ntypes}

    node_id_mapping = {nt: {old_i: new_i for new_i, old_i in enumerate(type_nodes[nt])} for nt in ntypes}
    het_dict = {}
    for u, v in list(G.edges()):
        nt1, nt2 = node_types[u], node_types[v]
        if nt1 not in ntypes or nt2 not in ntypes:
            continue

        if (nt1, nt2) == ('Court', 'Case'):
            nt2, nt1 = (nt1, nt2)
            u, v = (v, u)

        et = nt1[0] + nt2[0]
        if (nt1, et, nt2) not in het_dict:
            het_dict[(nt1, et, nt2)] = []
        e = (node_id_mapping[nt1][u], node_id_mapping[nt2][v])
        het_dict[(nt1, et, nt2)].append(e)

    refs_g = dgl.heterograph(het_dict, num_nodes_dict=num_nodes_dict)

    case_df, law_df, court_df = construct_data_df('Case'), construct_data_df('Law'), construct_data_df('Court')
    case_court_df = court_df.iloc[refs_g.edges(etype=('Case', 'CC', 'Court'))[1]]
    case_df = pd.concat([case_df, case_court_df.add_prefix('court_').reset_index(drop=True)], axis=1)

    return refs_g.node_type_subgraph(['Case', 'Law']), case_df, law_df


def ensure_law_data(path="./data/gesetze-im-internet",
                    repo_url="https://github.com/QuantLaw/gesetze-im-internet.git",
                    branch="data", revision="67e0567"):
    """
    Clones the specific 'data' branch and checks out a revision.
    """
    # Create the parent directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if not os.path.exists(os.path.join(path, ".git")):
        print(f"--- Data not found. Cloning branch '{branch}' from {repo_url} ---")
        # -b specifies the branch, --single-branch saves disk space/time
        subprocess.run(["git", "clone", "-b", branch, "--single-branch", repo_url, path], check=True)
    else:
        print(f"--- Repository exists at {path}. Fetching updates ---")
        subprocess.run(["git", "-C", path, "fetch", "origin"], check=True)

    print(f"--- Setting data to revision: {revision} ---")
    # This works whether revision is a branch name or a specific commit hash
    subprocess.run(["git", "-C", path, "checkout", revision], check=True)
    print("--- Law data ready ---")


def parse_german_laws(data_path: str = 'data/gesetze-im-internet/data/items'):
    documents = []

    all_xml_files = []
    for root, _, files in os.walk(data_path):
        for file in files:
            if file.endswith(".xml"):
                all_xml_files.append(os.path.join(root, file))

    for xml_path in tqdm(all_xml_files, desc="Parsing Laws", unit="file"):
        try:
            # Use recover=True in case of slightly malformed XML
            parser = etree.XMLParser(recover=True)
            tree = etree.parse(xml_path, parser=parser)

            filename = os.path.basename(xml_path)

            # German laws use <norm> tags for individual sections (§)
            norms = tree.xpath("//norm")

            for norm in norms:
                # 1. Safely extract metadata fields using xpath and ternary operators
                # .xpath returns a list; we check if it's non-empty before accessing index 0
                paragraph_list = norm.xpath(".//enbez/text()")
                title_list = norm.xpath(".//titel/text()")
                jurabk_list = norm.xpath(".//jurabk/text()")

                paragraph = paragraph_list[0].strip() if paragraph_list else ""
                title = title_list[0].strip() if title_list else ""
                law_book = jurabk_list[0].strip() if jurabk_list else "unknown"

                # 2. Extract content
                content = " ".join(norm.xpath(".//text//text()")).strip()

                # 3. Only add if there is actual text to index
                if content:
                    # Create a clean header for the RAG to "read"
                    # Example: "BGB § 2377: Wiederaufleben erloschener Rechtsverhältnisse"
                    full_title = f"{law_book} {paragraph}: {title}".strip(": ")

                    documents.append({
                        "text": content,
                        "title": full_title,
                        "filename": filename,
                        "law_book": law_book,
                        "paragraph": paragraph if paragraph else "unknown",
                        "source_path": xml_path
                    })
        except Exception as e:
            print(f"Error parsing {xml_path}: {e}")

    return documents
