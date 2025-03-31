# BMKG
Project Overview

This project comprises two major components: (1) the construction pipeline for a Biomedical Knowledge Graph (BMKG) and (2) a deep graph learning-based link prediction model. The system is designed to:

Construct large-scale knowledge graphs from biomedical data sources.

Perform efficient link prediction tasks.

Facilitate the discovery and inference of biomedical relationships.

Key Features:
Knowledge Graph Construction

Integration of heterogeneous biomedical data (e.g., PubMed abstracts, DrugBank, CTD)

Entity normalization and relation extraction with quality validation

Neo4j-based graph storage and semantic enrichment

Link Prediction Framework

Implementation of state-of-the-art graph neural networks (GCN, GAT, RGCN) and knowledge graph embedding methods (TransE, RotatE)

Optimized negative sampling strategies for biomedical relations

End-to-end training and evaluation pipelines
