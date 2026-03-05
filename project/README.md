# Genomic Network - Integrazione Espressione Genica
---

Questo progetto implementa il sistema di integrazione dei valori di espressione genica all'interno della **Genomic Network** basata sui dati tumorali TCGA. Il sistema permette di processare dataset massivi in formato TSV (mutazioni e matrici di espressione) per generare nodi e archi pronti per l'importazione in **Neo4j**.

L’implementazione è progettata per essere:
- **Scalabile**: Lettura in streaming delle matrici di espressione per ottimizzare l'uso della RAM.
- **Parallela**: Parsing delle mutazioni velocizzato tramite elaborazione multi-processo per cromosoma.
- **Riproducibile**: Utilizzo di una cache SQLite locale per il mapping degli Ensembl ID.
- **Validata**: Sistema integrato di controllo qualità degli output.
---
## Struttura del Progetto

```text
project/
├── src/
│   ├── main.py                # Entry point della riga di comando
│   ├── case1/                 # Logica specifica per il Caso 1
│   ├── case2/                 # Logica specifica per il Caso 2
│   └── datasets/              # Directory radice dei dati di input
│       ├── mutations/         # File chr_*.tsv (mutazioni per cromosoma)
│       ├── tumors/            # Matrici STAR-FPKM (es. TCGA-OV.star_fpkm.tsv)
│       ├── mappings/          # File symbol_to_ensg.tsv per risoluzione offline
│       ├── patients/          # Anagrafica pazienti (patients.tsv)
│       ├── diseases/          # Elenco patologie (diseases.tsv)
│       └── edges/             # Relazioni pre-esistenti (Paziente-Malattia, ecc.)
```
---

## Modalità di Integrazione

Il progetto supporta due modalità di modellazione del grafo:

### Caso 1: Nodi ExpressionProfiling
Crea un nodo `ExpressionProfiling` univoco per ogni coppia Gene-Paziente tramite hash MD5.
- **Logica**: `(:Patient) <-[:HAS_PROFILE]- (:ExpressionProfiling) -[:AFFECTS_GENE]-> (:Mutation)`
- **Output**: 
    - `expression_profiling_nodes.tsv`: Nodi con proprietà `gene_name` e `expression_value`.
    - `patient_expression_edges.tsv`: Archi tra paziente e profilo.
    - `expression_mutation_edges.tsv`: Archi tra profilo e mutazioni correlate.

### Caso 2: Nodi Gene e Relazioni Dirette
Crea un nodo `Gene` e inserisce il valore di espressione direttamente sull'arco verso il paziente.
- **Logica**: `(:Patient) -[:EXPRESSES {value: float}]-> (:Gene) -[:HAS_MUTATION]-> (:Mutation)`
- **Output**:
    - `genes.tsv`: Nodi `Gene` con proprietà `gene_id` e `chromosome`.
    - `patients_genes_expression_edges.tsv`: Archi **Patient → Gene** con proprietà `expression_value`.
    - `genes_mutations_edges.tsv`: Archi **Gene → Mutation**.

---

## Struttura Dataset Attesa
Il modulo elabora i file posizionati all'interno della directory `datasets`, aspettandosi la seguente alberatura:

```text
datasets/
├── mutations/
│   ├── chr_1.tsv
│   ├── chr_2.tsv
│   └── ... (fino a chr_Y.tsv)
├── tumors/
│   └── [nome_cancro].star_fpkm.tsv  # Matrice espressione (es. TCGA-OV (STAR-FPKM))
├── mappings/
│   └── symbol_to_ensg.tsv            # Mapping Ensembl ID -> Gene Symbol
├── patients/
│   └── patients.tsv                  # Anagrafica pazienti
├── diseases/
│   └── diseases.tsv                  # Elenco patologie
├── indexes/
│   └── mutation_index.tsv            # Indice delle mutazioni
└── edges/
    ├── patients_diseases/
    │   └── patients_diseases.tsv     # Relazioni Paziente-Malattia
    └── chr_1/ ... chr_Y/             # Relazioni Paziente-Mutazione per cromosoma
```
---
## Dettagli Implementativi

* **Risoluzione Geni (Offline Mapping)**: I geni nella matrice di espressione sono indicati per Ensembl ID. Il sistema implementa una risoluzione locale (`OfflineGeneResolver`) tramite mapping TSV e cache SQLite per convertire in modo efficiente gli identificativi nel nome del gene (considerando solo *Homo sapiens*).
* **Multiprocessing**: Il processamento dei file mutazioni divisi per cromosoma (`chr_*.tsv`) avviene in parallelo (tramite `ProcessPoolExecutor` configurabile con il parametro `workers`) per ottimizzare i tempi di estrazione della lista dei geni e degli archi gene-mutazione.
* **Validazione Automatica**: Al termine della generazione, il parametro `validate=True` permette alla pipeline di validare la correttezza degli header, verificare i valori numerici nel campo espressione e controllare la consistenza dei `mutation_id` estratti rispetto a quelli dei dataset originali.

## Esecuzione da linea di comando

Per avviare la pipeline da riga di comando, posizionarsi nella cartella `project/src` ed eseguire:
### **Caso 1:**
```bash
python .\main.py --case 1 --workers 8 --verbose true --validate true
```
Ecco una **documentazione CLI breve e pulita in Markdown** per il tuo `main.py`, aggiornata con gli argomenti e con `{}` per obbligatori e `[]` per opzionali.

### Caso 2
```bash
python main.py --case {2} [--workers N] [--verbose true|false] [--log_file true|false] [--validate true|false] [--tumors_glob PATTERN] [--tumor_files FILE1,FILE2,...] [--prefix_patient_id true|false] [--tumor_id_split_on CHAR]
```

### Parametri

| Argomento                       | Descrizione                                                                            |                                                                                             |
| ------------------------------- | -------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| `--case {2}`                    | Specifica quale caso eseguire (in questo caso la pipeline del **Caso 2**).             |                                                                                             |
| `--workers N`                   | Numero di processi paralleli per l’indicizzazione delle mutazioni (default: `1`).      |                                                                                             |
| `--verbose true                 | false`                                                                                 | Abilita/disabilita il logging su console (default: `true`).                                 |
| `--log_file true                | false`                                                                                 | Scrive il log anche su file nella cartella `output/tmp` (default: `false`).                 |
| `--validate true                | false`                                                                                 | Esegue la validazione degli output al termine della pipeline (default: `true`).             |
| `--tumors_glob PATTERN`         | Pattern per selezionare i file tumorali in `datasets/tumors/` (default: `*.tsv`).      |                                                                                             |
| `--tumor_files FILE1,FILE2,...` | Lista esplicita di file tumorali da processare (se non specificato usa `tumors_glob`). |                                                                                             |
| `--prefix_patient_id true       | false`                                                                                 | Prefissa `patient_id` con `tumor_id::` per evitare collisioni tra tumori (default: `true`). |
| `--tumor_id_split_on CHAR`      | Carattere usato per estrarre `tumor_id` dal nome del file tumorale (default: `.`).     |                                                                                             |

### Esempio

Esecuzione standard con 4 worker e validazione:

```bash
python main.py --case 2 --workers 4 --verbose true --validate true
```

### Selezionare solo alcuni tumori

```bash
python main.py --case 2 --tumor_files TCGA-BRCA.star_fpkm.tsv,TCGA-OV.star_fpkm.tsv
```

### Usare solo tumori con un certo pattern

```bash
python main.py --case 2 --tumors_glob "TCGA-*.tsv"
```

### Output generati

La pipeline produce nella cartella `output/`:

* `genes.tsv` → lista dei geni coinvolti nelle mutazioni
* `genes_mutations_edges.tsv` → relazioni gene–mutazione
* `patients_genes_expression_edges.tsv` → espressione genica per paziente (multi-tumor)

Formato di `patients_genes_expression_edges.tsv`:

```
tumor_id    patient_id    gene_id    expression_value
```

Ogni riga rappresenta l’espressione di un gene per un paziente in un determinato tumore.


Argomenti:
* `--case {1,2}`: esegue il caso richiesto (al momento implementato **Case 2**)
* `--verbose {true,false}`: log su console (se `false`, nessun log in console)
* `--workers N`: numero di processi per lo step mutazioni (parallelismo reale)
* `--validate {true,false}`: Abilita il controllo post-generazione (integrità degli ID MD5, header TSV e coerenza referenziale).
---

## Struttura dell'Output 
Una volta completata l'esecuzione, la struttura delle cartelle generata sarà la seguente:

### Caso 1
```text
src\output_case1
│   expression_mutation_edges.tsv
│   expression_profiling_nodes.tsv
│   patient_expression_edges.tsv
│
└───tmp/
        ensembl_cache_case1.db
        *.part.tsv
```

### Caso 2
```
src/output/
├── genes.tsv
├── genes_mutations_edges.tsv
├── patients_genes_expression_edges.tsv
└── tmp/
   ├── ensembl_cache.db
   └── parts/                             # part files generati dai worker (mutations)

```