#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash check_case2_output.sh
# Optional:
#   OUT_DIR=/path/to/out bash check_case2_output.sh

OUT_DIR="${OUT_DIR:-out}"
FINAL_DIR="${OUT_DIR%/}/final"

err() { echo "ERROR: $*" >&2; exit 1; }
info() { echo "[INFO] $*"; }

[ -d "$FINAL_DIR" ] || err "Final directory not found: $FINAL_DIR"

PAT_N="$FINAL_DIR/patients_nodes.tsv"
DIS_N="$FINAL_DIR/diseases_nodes.tsv"
MUT_N="$FINAL_DIR/mutations_nodes.tsv"
GEN_N="$FINAL_DIR/genes_nodes.tsv"

E_PD="$FINAL_DIR/edges_patient_disease.tsv"
E_PM="$FINAL_DIR/edges_patient_mutation.tsv"
E_PG="$FINAL_DIR/edges_patient_gene_expression.tsv"
E_GM="$FINAL_DIR/edges_gene_mutation.tsv"

for f in "$PAT_N" "$DIS_N" "$MUT_N" "$GEN_N" "$E_PD" "$E_PM" "$E_PG" "$E_GM"; do
  [ -f "$f" ] || err "Missing file: $f"
done

info "Checking headers..."
hp="$(head -n 1 "$PAT_N")"
hd="$(head -n 1 "$DIS_N")"
hm="$(head -n 1 "$MUT_N")"
hg="$(head -n 1 "$GEN_N")"
hpd="$(head -n 1 "$E_PD")"
hpm="$(head -n 1 "$E_PM")"
hpg="$(head -n 1 "$E_PG")"
hgm="$(head -n 1 "$E_GM")"

echo "patients_nodes.tsv:                $hp"
echo "diseases_nodes.tsv:                $hd"
echo "mutations_nodes.tsv:               $hm"
echo "genes_nodes.tsv:                   $hg"
echo "edges_patient_disease.tsv:         $hpd"
echo "edges_patient_mutation.tsv:        $hpm"
echo "edges_patient_gene_expression.tsv: $hpg"
echo "edges_gene_mutation.tsv:           $hgm"
echo

info "Quick sanity: header duplication check (PM edge) ..."
pm2="$(sed -n '2p' "$E_PM" || true)"
if [ "$pm2" = "$hpm" ]; then
  err "edges_patient_mutation.tsv has duplicated header (line 2 equals header)."
else
  info "OK: no duplicated header in edges_patient_mutation.tsv"
fi
echo

info "Line counts (wc -l):"
wc -l "$FINAL_DIR"/*.tsv | sed 's#'"$FINAL_DIR/"'##'
echo

# ------------------------------
# Build unique ID sets
# ------------------------------
TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

info "Building unique ID sets (may take a bit on large edge files)..."

cut -f1 "$PAT_N" | tail -n +2 | sort -u > "$TMPDIR/p_nodes.txt"
cut -f1 "$MUT_N" | tail -n +2 | sort -u > "$TMPDIR/m_nodes.txt"
cut -f1 "$GEN_N" | tail -n +2 | sort -u > "$TMPDIR/g_nodes.txt"
cut -f1 "$DIS_N" | tail -n +2 | sort -u > "$TMPDIR/d_nodes.txt"

# Detect edge column positions from headers (1-based)
# We support either order (patient_id/mutation_id, mutation_id/patient_id, etc.)
col_index() {
  local header="$1"
  local name="$2"
  # print field index
  awk -v hdr="$header" -v tgt="$name" 'BEGIN{
    n=split(hdr,a,"\t");
    for(i=1;i<=n;i++){ if(a[i]==tgt){ print i; exit 0; } }
    exit 1
  }'
}

pm_p_idx="$(col_index "$hpm" "patient_id" || true)"
pm_m_idx="$(col_index "$hpm" "mutation_id" || true)"
[ -n "${pm_p_idx:-}" ] && [ -n "${pm_m_idx:-}" ] || err "Cannot locate patient_id/mutation_id in PM header: $hpm"

pd_p_idx="$(col_index "$hpd" "patient_id" || true)"
pd_d_idx="$(col_index "$hpd" "disease_id" || true)"
[ -n "${pd_p_idx:-}" ] && [ -n "${pd_d_idx:-}" ] || err "Cannot locate patient_id/disease_id in PD header: $hpd"

pg_p_idx="$(col_index "$hpg" "patient_id" || true)"
pg_g_idx="$(col_index "$hpg" "gene_id" || true)"
pg_e_idx="$(col_index "$hpg" "expression_value" || true)"
[ -n "${pg_p_idx:-}" ] && [ -n "${pg_g_idx:-}" ] && [ -n "${pg_e_idx:-}" ] || err "Cannot locate patient_id/gene_id/expression_value in PG header: $hpg"

gm_g_idx="$(col_index "$hgm" "gene_id" || true)"
gm_m_idx="$(col_index "$hgm" "mutation_id" || true)"
[ -n "${gm_g_idx:-}" ] && [ -n "${gm_m_idx:-}" ] || err "Cannot locate gene_id/mutation_id in GM header: $hgm"

# Create unique IDs used in edges
cut -f"$pd_p_idx" "$E_PD" | tail -n +2 | sort -u > "$TMPDIR/p_in_pd.txt"
cut -f"$pd_d_idx" "$E_PD" | tail -n +2 | sort -u > "$TMPDIR/d_in_pd.txt"

cut -f"$pm_p_idx" "$E_PM" | tail -n +2 | sort -u > "$TMPDIR/p_in_pm.txt"
cut -f"$pm_m_idx" "$E_PM" | tail -n +2 | sort -u > "$TMPDIR/m_in_pm.txt"

cut -f"$pg_p_idx" "$E_PG" | tail -n +2 | sort -u > "$TMPDIR/p_in_pg.txt"
cut -f"$pg_g_idx" "$E_PG" | tail -n +2 | sort -u > "$TMPDIR/g_in_pg.txt"

cut -f"$gm_g_idx" "$E_GM" | tail -n +2 | sort -u > "$TMPDIR/g_in_gm.txt"
cut -f"$gm_m_idx" "$E_GM" | tail -n +2 | sort -u > "$TMPDIR/m_in_gm.txt"

echo
info "Integrity checks (any output below means missing references):"

echo "---- Missing patients referenced in edges_patient_disease ----"
comm -23 "$TMPDIR/p_in_pd.txt" "$TMPDIR/p_nodes.txt" | head

echo "---- Missing diseases referenced in edges_patient_disease ----"
comm -23 "$TMPDIR/d_in_pd.txt" "$TMPDIR/d_nodes.txt" | head

echo "---- Missing patients referenced in edges_patient_mutation ----"
comm -23 "$TMPDIR/p_in_pm.txt" "$TMPDIR/p_nodes.txt" | head

echo "---- Missing mutations referenced in edges_patient_mutation ----"
comm -23 "$TMPDIR/m_in_pm.txt" "$TMPDIR/m_nodes.txt" | head

echo "---- Missing patients referenced in edges_patient_gene_expression ----"
comm -23 "$TMPDIR/p_in_pg.txt" "$TMPDIR/p_nodes.txt" | head

echo "---- Missing genes referenced in edges_patient_gene_expression ----"
comm -23 "$TMPDIR/g_in_pg.txt" "$TMPDIR/g_nodes.txt" | head

echo "---- Missing genes referenced in edges_gene_mutation ----"
comm -23 "$TMPDIR/g_in_gm.txt" "$TMPDIR/g_nodes.txt" | head

echo "---- Missing mutations referenced in edges_gene_mutation ----"
comm -23 "$TMPDIR/m_in_gm.txt" "$TMPDIR/m_nodes.txt" | head
echo

info "Connectivity sanity: patients that appear in BOTH expression and mutation edges (show up to 20)"
comm -12 "$TMPDIR/p_in_pg.txt" "$TMPDIR/p_in_pm.txt" | head -n 20
echo

info "Done."