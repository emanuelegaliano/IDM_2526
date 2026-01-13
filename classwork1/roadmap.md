## Pipeline di progetto: fasi e flussi di elaborazione

Il progetto è strutturato come una sequenza di **pipeline modulari**, implementate tramite la libreria **TRACCIA**, in cui ogni fase corrisponde a uno specifico task della traccia.
Ogni pipeline opera su uno stato condiviso (`Footprint`) e produce output intermedi o finali utilizzabili dalle fasi successive.

---

### **Fase A — Setup e inizializzazione**

La fase iniziale definisce l’infrastruttura del progetto e prepara gli elementi comuni a tutte le pipeline successive.
In questa fase vengono istanziati:

* l’oggetto di configurazione (`Config`), contenente parametri globali e soglie;
* il `Footprint`, che funge da contenitore per dati, risultati intermedi e metriche;
* le dipendenze necessarie all’esecuzione delle pipeline.

**Output della fase:** ambiente pronto, configurazione centralizzata e footprint inizializzato.

---

### **Fase B — Data loading e cleaning**

Questa pipeline ha l’obiettivo di leggere il dataset grezzo e produrre una versione pulita e coerente dei dati.

La pipeline comprende i seguenti step:

1. caricamento del dataset;
2. normalizzazione dello schema delle colonne;
3. parsing e combinazione delle variabili temporali (data e ora);
4. rimozione delle righe non valide;
5. esclusione dei record relativi agli “shoppers” secondo una regola configurabile;
6. aggiunta delle variabili temporali derivate (periodi dell’anno e fasce orarie).

**Output della fase:** `clean_df`, dataset pulito e arricchito, utilizzato come input per tutte le fasi successive.

---

### **Fase C — Task 1: frequenze per livello di merchandising**

Questa pipeline implementa il Task 1 della traccia e calcola le frequenze degli elementi per ciascun livello della gerarchia di merchandising (liv1–liv4).

La pipeline:

1. calcola le tabelle di frequenza per ogni livello;
2. seleziona i top-5 e bottom-5 elementi;
3. genera e salva i corrispondenti barplot.

**Output della fase:**

* tabelle di frequenza per livello;
* plot top/bottom per ciascun livello.

---

### **Fase D — Task 2: frequenze stratificate**

Questa pipeline estende il Task 1 introducendo una stratificazione temporale delle frequenze.

L’elaborazione viene ripetuta:

* per tre periodi dell’anno;
* per tre fasce orarie della giornata.

La pipeline riutilizza la logica di calcolo delle frequenze della Fase C, applicandola separatamente a ciascuno strato temporale.

**Output della fase:**

* tabelle di frequenza stratificate;
* plot top/bottom per ogni combinazione livello–strato.

---

### **Fase E — Task 3 e Task 4: Association Rules**

Questa fase comprende due pipeline distinte per l’estrazione di regole di associazione a livello 4:

* una pipeline basata su **Apriori**;
* una pipeline basata su **FP-Growth**.

Le pipeline includono:

1. costruzione delle transazioni a livello di scontrino;
2. codifica one-hot delle transazioni;
3. mining delle regole secondo le soglie configurate;
4. salvataggio delle regole estratte.

**Output della fase:**

* file contenenti le regole di associazione con support, confidence e lift;
* metriche riassuntive sul numero e sulla qualità delle regole.

---

### **Fase F — Task 5: PCA e clustering delle tessere**

L’ultima pipeline realizza il Task 5 e si concentra sulla segmentazione dei clienti.

La pipeline:

1. filtra le transazioni associate a tessere valide;
2. costruisce la matrice tessera × prodotto con frequenze di acquisto;
3. normalizza i dati;
4. applica la PCA mantenendo una quota prefissata di varianza;
5. esegue il clustering nello spazio ridotto;
6. valuta la qualità del clustering tramite metriche quantitative.

**Output della fase:**

* embedding PCA;
* assegnazioni di cluster per tessera;
* metriche di qualità del clustering.