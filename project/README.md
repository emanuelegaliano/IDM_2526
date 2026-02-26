# Progetto Data Mining (APPUNTI)

Questo progetto implementa il **Case 2** dell’assignment: a partire da dataset TSV (mutations + tumors expression) genera nodi e archi in formato TSV, pronti per essere importati in un grafo (o utilizzati per analisi successive).

L’implementazione è pensata per essere:
- **scalabile in RAM** (lettura a streaming dei TSV)
- **parallela** sul parsing delle mutazioni (per cromosoma)
- **riproducibile** (cache SQLite per mapping Ensembl)
- **validata** automaticamente (se abilitato)

---

## Struttura del progetto

```

project/
├── README.md
└── src/
├── main.py
├── case2/
│   ├── **init**.py
│   ├── config.py
│   ├── pipeline.py
│   ├── gene_resolver.py
│   └── validate_outputs.py
└── datasets/
├── diseases/                # (fisso) diseases.tsv
├── patients/                # (fisso) patients.tsv
├── edges/
│   ├── chr_*/               # edges pazienti-mutazioni (opzionali per questo step)
│   └── patients_diseases/   # patients_diseases.tsv
├── mutations/               # chr_*.tsv (mutazioni per cromosoma)
└── tumors/                  # matrice di espressione (TSV, es. TCGA-OV.star_fpkm.tsv)

```

> Nota: `diseases.tsv` e `patients.tsv` sono presenti come dataset “fissi” del progetto, ma la pipeline Case 2 corrente genera principalmente i file richiesti per nodi/archi legati a **mutations** e **tumor expression**.

---

## Cosa fa il Case 2

La pipeline del Case 2 esegue questi step:

### Step 1 — Mutations (parallelo per cromosoma)
Legge i file `src/datasets/mutations/chr_*.tsv` e costruisce:

- `genes_mutations_edges.tsv`  
  archi **Gene → Mutation**:
  - `gene_id` (Gene.refGene, in genere SYMBOL)
  - `mutation_id` (unique_id)

In parallelo, raccoglie anche la whitelist dei geni osservati e il cromosoma d’origine.

### Step 2 — Nodes Gene
Genera `genes.tsv` con:
- `gene_id`
- `chromosome` (es. `22` oppure lista `1,22,X` se un gene appare in più cromosomi)

### Step 3 — Tumor expression (streaming)
Legge la matrice `src/datasets/tumors/*.tsv` (prima colonna gene, colonne successive pazienti) e genera:

- `patients_genes_expression_edges.tsv`  
  archi **Patient → Gene**:
  - `patient_id`
  - `gene_id`
  - `expression_value`

#### Mapping Ensembl → Symbol (per allineare gene_id)
Nel dataset TCGA (es. UCSC Xena) la prima colonna spesso è `Ensembl_ID` (tipo `ENSG...` con `.version`).
Le mutazioni invece sono tipicamente in gene symbol.

Per far combaciare i `gene_id`, il progetto supporta:

- `gene_id_mode="ensembl_api_cache"`:  
  - risolve **solo la whitelist** dei gene symbol verso `ENSG...` tramite Ensembl REST
  - crea una mappa `ENSG -> SYMBOL` (in cache SQLite)
  - streamma la matrice e converte `ENSG(.version)` in `SYMBOL`

Questo evita chiamate API per ogni riga della matrice ed è scalabile.

---

## Output

Quando esegui `--case 2`, l’output viene scritto in:

```

src/output/
├── genes.tsv
├── genes_mutations_edges.tsv
├── patients_genes_expression_edges.tsv
└── tmp/
├── ensembl_cache.db
└── parts/   # part files generati dai worker (mutations)

```

---

## Esecuzione da linea di comando

Dalla cartella `src/`:

### Caso 2
```bash
python main.py --case 2 --verbose true --log_file false --workers 4
````

Argomenti:

* `--case {1,2}`: esegue il caso richiesto (al momento implementato **Case 2**)
* `--verbose {true,false}`: log su console (se `false`, nessun log in console)
* `--log_file {true,false}`: log su file (in `src/output/case2.log` se abilitato)
* `--workers N`: numero di processi per lo step mutazioni (parallelismo reale)

---

## Validazione output (opzionale)

La pipeline può validare automaticamente i file generati.

* In `Case2Config` esiste `validate: bool`.
* Se `validate=True`, a fine pipeline esegue `validate_outputs.py` e:

  * controlla header TSV
  * controlla coerenza `gene_id` (edges ⊆ nodes)
  * controlla coerenza `mutation_id` (edges ⊆ `unique_id` delle mutazioni)
  * sanity check numerico su `expression_value` (campionamento)

Se fallisce, la pipeline termina con **exit code 1**.

---

## Note su performance

* Parsing mutazioni: parallelizzato per cromosoma con `ProcessPoolExecutor`.
* Matrice di espressione: letta a streaming (una riga gene per volta).
* Ensembl REST: con `gene_id_mode="ensembl_api_cache"` le chiamate sono limitate alla whitelist (tipicamente centinaia, non decine di migliaia) e vengono **cache-ate** in `output/tmp/ensembl_cache.db`.

### Rate limiting (HTTP 429)

Con molti thread Ensembl può rispondere `429 Too Many Requests`. Il client:

* fa retry con backoff
* rispetta `Retry-After` se presente
* cache-a i risultati: nelle run successive molte chiamate spariscono

---

## Troubleshooting

### “field larger than field limit”

La pipeline alza `csv.field_size_limit` automaticamente, quindi non dovrebbe comparire più.

### Nessun match tra genes e expression

Assicurati di usare:

* `gene_id_mode="ensembl_api_cache"`
  se mutazioni sono SYMBOL e la matrice è ENSG.

### Troppi 429 da Ensembl

Riduci i thread della whitelist mapping (parametro interno in `pipeline.py`) oppure lancia una prima run con `workers` basso, poi sfrutta la cache.

---

## Public API (package `case2`)

Nel package `case2` sono esportati:

* `Case2Config`
* `run_case2`
* `validate_case2_outputs` (standalone)

Esempio (uso programmatico):

```python
from pathlib import Path
from case2 import Case2Config, run_case2

cfg = Case2Config(
    datasets_root=Path("datasets"),
    output_dir=Path("output"),
    tmp_dir=Path("output/tmp"),
    workers=4,
    verbose=True,
    log_to_file=False,
    gene_id_mode="ensembl_api_cache",
    validate=True,
)

run_case2(cfg)
```