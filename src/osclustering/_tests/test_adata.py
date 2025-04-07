import higra as hg
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import scipy.cluster.hierarchy as sch
from matplotlib import colormaps
from matplotlib.colors import to_hex

from osclustering import optimal_stability_clustering
from osclustering._adata import dropout_counts
from osclustering._graph import knn_mst_graph


def _pre_process_counts(adata: sc.AnnData) -> sc.AnnData:
    """
    Pre-process counts to prepare for clustering.
    """
    adata = adata.copy()
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)

    sc.pp.highly_variable_genes(adata)

    adata.raw = adata.copy()
    adata = adata[:, adata.var.highly_variable]

    sc.pp.pca(adata)

    n_neighbors = 10
    n_pcs = 40
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)

    adata.obsp["knn_mst_distances"] = knn_mst_graph(
        adata.obsm["X_pca"][:, :n_pcs],
        n_neighbors=n_neighbors,
    )

    adata.uns["knn_mst_neighbors"] = dict(
        n_neighbors=n_neighbors,
        n_pcs=n_pcs,
        distances_key="knn_mst_distances",
    )

    return adata


def test_pbmc3k(
    show: bool = False,
) -> None:
    # updated legacy workflow https://scanpy.readthedocs.io/en/stable/tutorials/basics/clustering-2017.html
    # based on https://scanpy.readthedocs.io/en/stable/tutorials/basics/clustering.html
    adata = sc.datasets.pbmc3k()

    # mitochondrial genes, "MT-" for human, "Mt-" for mouse
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    # ribosomal genes
    adata.var["ribo"] = adata.var_names.str.startswith(("RPS", "RPL"))
    # hemoglobin genes
    adata.var["hb"] = adata.var_names.str.contains("^HB[^(P)]")

    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt", "ribo", "hb"], inplace=True, log1p=True
    )
    if show:
        sc.pl.violin(
            adata,
            ["n_genes_by_counts", "total_counts", "pct_counts_mt"],
            jitter=0.4,
            multi_panel=True,
        )
        plt.show()

    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)

    sc.pp.scrublet(adata)

    adata.layers["counts"] = adata.X.copy()

    ref_adata = _pre_process_counts(adata)

    perturbed_adata = dropout_counts(adata, dropout_rate=0.2, n_replicates=10)
    perturbed_adata = [_pre_process_counts(adata) for adata in perturbed_adata]

    trees = []
    for data in [ref_adata, *perturbed_adata]:
        graph, weights = hg.adjacency_matrix_2_undirected_graph(
            data.obsp["knn_mst_distances"]
        )
        tree, _ = hg.watershed_hierarchy_by_number_of_parents(
            graph, weights, canonize_tree=False
        )
        trees.append(tree)

    os_clusters, os_weights = optimal_stability_clustering(
        trees[0],
        trees[1:],
        single_cluster_threshold=0.0,
        return_tree_weights=True,
    )

    cmap = colormaps["magma"]
    norm_os_weights = os_weights / (len(perturbed_adata) * hg.attribute_area(trees[0]))

    irrelevant_nodes = norm_os_weights < 0.8

    tree = trees[0]
    area = hg.attribute_area(tree)
    tree, node_map = hg.simplify_tree(tree, irrelevant_nodes)

    print("Num significant nodes", (~irrelevant_nodes).sum())

    binary_tree, binary_tree_node_map = hg.tree_2_binary_tree(tree)
    binary_tree_os_weights = norm_os_weights[node_map][binary_tree_node_map]
    binary_tree_area = area[node_map][binary_tree_node_map]

    linkage_matrix = hg.binary_hierarchy_to_scipy_linkage_matrix(
        binary_tree, binary_tree_area
    )

    sch.dendrogram(
        linkage_matrix,
        link_color_func=lambda x: to_hex(cmap(binary_tree_os_weights[x])),
        truncate_mode="level",
        p=15,
    )
    plt.show()

    ref_adata.obs["os_clusters"] = os_clusters
    ref_adata.obs["os_clusters"] = ref_adata.obs["os_clusters"].astype("category")

    columns = []

    for i, n in enumerate(range(tree.num_leaves(), tree.num_vertices() - 1)):
        _, sub_tree_node_map = tree.sub_tree(n)
        leaves = sub_tree_node_map[sub_tree_node_map < tree.num_leaves()]
        labels = np.zeros(tree.num_leaves(), dtype=bool)
        labels[leaves] = True
        col_name = f"os_cluster_{i}"
        ref_adata.obs[col_name] = labels
        ref_adata.obs[col_name] = ref_adata.obs[col_name].astype("category")
        columns.append(col_name)

    sc.tl.leiden(
        ref_adata,
        resolution=0.9,
        random_state=0,
        flavor="igraph",
    )

    sc.tl.umap(ref_adata)
    sc.pl.umap(ref_adata, color=["leiden", "os_clusters"] + columns, alpha=0.5)

    plt.show()


if __name__ == "__main__":
    """
    Run this test directly to ensure the pbmc3k dataset can be loaded.
    """
    test_pbmc3k()
