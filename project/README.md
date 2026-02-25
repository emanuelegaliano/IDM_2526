# Progetto – Gene Expression Integration

## Abstract Caso 2

Questo progetto implementa il **Case 2** del progetto di Data Mining:
integrare i dati di mutazione genomica TCGA con la matrice di espressione genica (STAR FPKM) proveniente da Xena, costruendo una struttura coerente pronta per l’import in Neo4j.

Nel Case 2 viene richiesto di:

* creare un nodo `Gene`
* collegare `Patient → Gene` con un arco contenente `expression_value`
* collegare `Gene → Mutation`
* mantenere la struttura originale `Patient → Disease` e `Patient → Mutation`

Tumore selezionato:
**OV – Ovarian Serous Cystadenocarcinoma**

Matrice di espressione:
[https://xenabrowser.net/datapages/?dataset=TCGA-OV.star_fpkm.tsv&host=https%3A%2F%2Fgdc.xenahubs.net](https://xenabrowser.net/datapages/?dataset=TCGA-OV.star_fpkm.tsv&host=https%3A%2F%2Fgdc.xenahubs.net)

---

## Struttura del Progetto

```
project/
├── main.py
├── src/
│   ├── case2/
│   │   ├── cfg.py
│   │   ├── check_dataset.py
│   │   ├── build_case2_intermediate.py
│   │   ├── build_case2_final.py
│   ├── datasets/
│   │   ├── chr_22.tsv
│   │   ├── chr_22_patients_mutations_edges.tsv
│   │   ├── patients.tsv
│   │   ├── diseases.tsv
│   │   ├── patients_diseases.tsv
│   │   ├── TCGA-OV.star_fpkm.tsv
├── out/
│   ├── generated/
│   └── final/
```

---

## Logica del Programma

Il progetto è diviso in tre fasi principali:

---

### 1️⃣ Verifica integrità dataset (`check_dataset.py`)

Controlla:

* coerenza delle foreign key:

  * `Patient → Disease`
  * `Patient → Mutation`
* duplicati negli edges
* mutazioni orfane
* pazienti senza collegamenti

Serve a garantire che il grafo sia consistente prima di integrare l’espressione.

---

### 2️⃣ Costruzione file intermedi (`build_case2_intermediate.py`)

1. Caricamento mutazioni per cromosomi configurati
2. Creazione nodo `Gene`

   * `gene_id = md5(gene_name + chromosome)`
3. Creazione archi:

   * `Gene → Mutation`
4. Lettura matrice di espressione
5. Trasformazione wide → long
6. Creazione archi:

   * `Patient → Gene` con `expression_value`

Output generati in:

```
out/generated/
```

* `genes.tsv`
* `gene_mutation_edges.tsv`
* `patient_gene_expression_edges.tsv`

---

### 3️⃣ Costruzione export finale (`build_case2_final.py`)

Unisce:

* nodi originali (Patient, Disease, Mutation)
* nodi Gene
* tutti gli edges

Filtra entità non referenziate.

Output finale pronto per Neo4j:

```
out/final/
```

* `patients_nodes.tsv`
* `diseases_nodes.tsv`
* `mutations_nodes.tsv`
* `genes_nodes.tsv`
* `edges_patient_disease.tsv`
* `edges_patient_mutation.tsv`
* `edges_patient_gene_expression.tsv`
* `edges_gene_mutation.tsv`

---

## Configurazione (`cfg.py`)

Tutta la configurazione è centralizzata in `src/case2/cfg.py`.

### Selezione cromosomi

```python
CHROMOSOMES = ["22"]
```

Per aggiungere cromosomi:

```python
CHROMOSOMES = ["22", "17"]
```

Il sistema genera automaticamente:

* lista file mutazioni
* lista file edges paziente-mutazione

---

### Schema colonne configurabili

Nel `cfg.py` è possibile specificare:

```python
MUTATION_ID_COL = "unique_id"
MUTATION_GENE_COL = "Gene.refGene"
MUTATION_CHR_COL = "Chr"
```

Override opzionali:

```python
PATIENT_ID_COL = None
EDGE_PM_MUTATION_COL = "mutation_id"
```

---

## Logging

Sistema di logging centralizzato in `cfg.py`.

Utilizzo negli script:

```python
import cfg
log = cfg.get_logger("nome_script")
```

Formato:

```
YYYY-MM-DD HH:MM:SS | LEVEL | module | message
```

Configurabile:

```python
LOG_LEVEL = logging.INFO
LOG_TO_FILE = False
```

---

## Output Atteso

Il risultato finale è un insieme coerente di file TSV pronti per:

* `neo4j-admin import`
* `LOAD CSV`
* oppure qualunque sistema di caricamento grafo

Il grafo finale contiene:

```
(Disease)
    ↑
    |
 (Patient)
    |        \
    ↓         \
 (Mutation)   (Gene)
                 ↑
                 |
           expression_value
```

---

## Test fatti

✔ Validazione integrità su:

* Patient → Disease
* Patient → Mutation

✔ Costruzione nodo Gene coerente con cromosoma

✔ Generazione Gene → Mutation

✔ Trasformazione matrice espressione (wide → long)

✔ Verifica overlap pazienti tra mutazioni ed espressione

✔ Filtraggio nodi non referenziati nel file finale

✔ Esecuzione su cromosoma 22 (~394k mutazioni)

---

## Stato attuale

Progetto funzionante per:

* 1 o più cromosomi
* 1 o più matrici di espressione
* configurazione centralizzata
* logging strutturato
* output separato in `out/`