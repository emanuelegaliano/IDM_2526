# ![IDM Logo](readme_logo.png)

# Introduction to Data Mining (IDM 2025/2026)

This repository collects **lecture notes**, **classwork notebooks**, and the **final project** developed for the course **“Introduction to Data Mining”** during the **2025/2026 academic year** at the **University of Catania**.

Most of the explanatory material is written in **Italian** (especially the notes and assignment descriptions), while several code artifacts, folder names, and implementation details follow a more **English-oriented** structure.

---

# Main Notes

The core theoretical material of the repository is the LaTeX manuscript:

```text
Dispense/DataMining.tex
```

which compiles into:

```text
Dispense/DataMining.pdf
```

The notes provide a structured overview of the main topics covered in the course, including:

* Mathematical prerequisites
* Data preprocessing
* Frequent itemsets
* Graphs and graph mining
* Subgraph matching
* Regression
* Classification
* Clustering
* Neural networks
* Transformer models, LLMs, and vector databases

The document is modularized in the `Dispense/chapters/` directory and supported by a large collection of figures inside `Dispense/images/`.

---

# Repository Structure

Below is the logical organization of the repository:

```text
.
├── Dispense/                  # LaTeX notes, compiled PDF, bibliography, images, front matter
├── classwork1/                # First classwork notebook + modular data-mining pipelines
├── classwork2/                # Second classwork notebook + ML experimentation scripts
├── project/                   # Final project on genomic network integration
├── src/                       # Auxiliary/generated data area at repository root
├── README.md
└── readme_logo.svg
```

## `Dispense/`

Contains the main lecture notes and related assets.

**Relevant files and folders:**
* `Dispense/DataMining.tex` — main LaTeX source
* `Dispense/DataMining.pdf` — compiled PDF
* `Dispense/chapters/` — chapter sources
* `Dispense/images/` — figures and illustrations
* `Dispense/frontmatter/` — prefatory and license material
* `Dispense/refs.bib` — bibliography

## `classwork1/`

The first classwork is organized around a notebook-driven workflow and a reusable pipeline structure based on **TRACCIA**.

### What it contains
* `classwork1/assignment.ipynb` — main notebook
* `classwork1/roadmap.md` — explanation of the project phases
* `classwork1/src/domain/` — configuration and shared state (`Footprint`)
* `classwork1/src/pipelines/` — modular processing stages
* `classwork1/src/utilities/` — helper functions for frequencies, plotting, and file handling

### What the classwork does
The notebook and pipelines cover a complete workflow that includes:

* data loading and cleaning
* schema normalization and temporal feature engineering
* frequency analysis across merchandising hierarchy levels
* stratified frequency analysis by time bands / seasonal partitions
* association rule mining with **Apriori** and **FP-Growth**
* PCA-based dimensionality reduction
* clustering of customer-card behavior in reduced space

## `classwork2/`

The second classwork focuses on a **multi-class classification** task and is organized as a sequence of explicit phases.

### What it contains
* `classwork2/assignment.ipynb` — main notebook
* `classwork2/src/phase0_clean.py` — dataset cleaning and consolidation
* `classwork2/src/phase1_pca2d.py` — 2D PCA projection and visualization
* `classwork2/src/phase2_gridsearch.py` — model selection with cross validation
* `classwork2/src/phase3_ensembles.py` — ensemble-oriented experimentation
* `classwork2/data/` — local input/intermediate data
* `classwork2/reports/` — generated artifacts such as plots and JSON summaries

### What the classwork does
The workflow includes:

* Excel-to-CSV cleaning and harmonization
* exploratory visualization with **PCA (2D)**
* stratified train/test split
* **GridSearchCV** over multiple models
* comparison of classifiers such as:
  * Decision Tree
  * Random Forest
  * Support Vector Classifier
  * k-NN
* export of metrics and experiment summaries

## `project/`

The final project implements a pipeline for integrating **gene expression** and **mutation** data into a **genomic network** intended for graph-based analysis and Neo4j import.

### What it contains
* `project/README.md` — project-specific documentation
* `project/src/main.py` — CLI entry point
* `project/src/case1/` — implementation of the first graph modeling strategy
* `project/src/case2/` — implementation of the second graph modeling strategy

### Project overview
The project supports two graph-construction strategies:

#### Case 1 — `ExpressionProfiling` nodes
Creates an intermediate expression node for each gene–patient pair and links it to both patients and mutations.

Main exported files:
* `expression_profiling_nodes.tsv`
* `patient_expression_edges.tsv`
* `expression_mutation_edges.tsv`

#### Case 2 — direct patient–gene expression relations
Builds `Gene` nodes and stores expression values directly on patient–gene edges, while keeping separate gene–mutation links.

Main exported files:
* `genes.tsv`
* `genes_mutations_edges.tsv`
* `patients_genes_expression_edges.tsv`

### Implementation highlights
The project includes:

* offline gene-symbol resolution through local TSV mapping + SQLite cache
* multiprocessing over mutation files and tumor matrices
* validation of generated TSV outputs
* a configurable CLI for selecting the execution case and runtime parameters

### Expected dataset layout
The project code expects the working datasets to be placed under:

```text
project/src/datasets/
```

with a structure consistent with the one described in `project/README.md` (mutations, tumors, mappings, patients, diseases, and edge files).

---

# Quick Start

## 1. Clone the repository

```bash
git clone https://github.com/emanuelegaliano/IDM_2526
cd IDM_2526
```

## 2. Compile the notes

From inside `Dispense/`:

```bash
latexmk -pdf -outdir=out DataMining.tex
```

This generates the compiled PDF and the auxiliary files in `Dispense/out/`.

## 3. Run the first classwork

```bash
pip install -r classwork1/requirements.txt
jupyter notebook classwork1/assignment.ipynb
```

## 4. Run the second classwork

```bash
pip install -r classwork2/requirements.txt
jupyter notebook classwork2/assignment.ipynb
```

## 5. Run the final project

Move into the project source directory and place the required datasets under `project/src/datasets/`, then run one of the supported cases.

### Case 1
```bash
cd project/src
python main.py --case 1 --workers 4 --verbose true --validate true
```

### Case 2
```bash
cd project/src
python main.py --case 2 --workers 4 --verbose true --validate true
```

### Optional arguments for Case 2
```bash
python main.py --case 2 --tumor_files TCGA-BRCA.star_fpkm.tsv,TCGA-OV.star_fpkm.tsv
python main.py --case 2 --tumors_glob "TCGA-*.tsv"
python main.py --case 2 --prefix_patient_id true --tumor_id_split_on "."
```

More details in the **[Project Readme](/project/README.md)**.

---

# How to Compile

The theoretical notes are written in LaTeX and rely on a standard academic toolchain.

## Requirements
* A LaTeX distribution (TeX Live, MiKTeX, etc.)
* `latexmk`
* `bibtex`

## Compile the document
From `Dispense/`, run:

```bash
latexmk -pdf -outdir=out DataMining.tex
```

## Clean auxiliary files
A simple way to remove build artifacts is:

```bash
rm -rf Dispense/out/*
```

---

# Contributing

Contributions are very welcome.

You can contribute by:

* Opening **issues** (typos, conceptual mistakes, broken commands, documentation improvements)
* Submitting **pull requests**
* Improving the notes, notebooks, figures, or project documentation
* Reporting reproducibility issues in the classworks or final project

## Pull Request Requirements

If you submit a pull request, please ensure that:

1. ✅ You modify the **source files** (`.tex`, `.ipynb`, `.py`, `.md`), not only exported artifacts
2. ✅ You verify that the affected section still runs or compiles correctly
3. ✅ You update generated files only when relevant to the change
4. ❗ You do **not** alter the repository structure unless strictly necessary
5. ✅ You keep code and documentation changes coherent with the repository style
6. ✅ You add your name to the **Contributors** section at the end of this README

Pull requests that do not meet these requirements may not be accepted.

---

# Authors

* **[Emanuele Galiano](https://github.com/emanuelegaliano)**

---

# Contributors

Add your name here when contributing through an accepted pull request.

---

# License

See `Dispense/frontmatter/license.tex` for license details.