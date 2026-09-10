Now I have the full picture. Here's what answers your two questions.

## Where to find "how well the unlearning happened" and "how well the model still works elsewhere"

The primary file is **[summary.csv](4methods/summary.csv)** — one row per `architecture` × `seed` × `forgotten_domain` × `method`, with `original`/`method`/`retrained` columns side by side for easy comparison. `retrained` is the "gold" retain-only model, i.e. the ideal outcome of unlearning. Use **[summary_aggregated.csv](4methods/summary_aggregated.csv)** for the same thing averaged across seeds (mean/std/CI).

### 1. How well the unlearning happened (forgetting quality)
The notebook's own interpretation note is explicit: *"The correct target is closeness to retained-only retraining ... A membership AUC near 0.5 is not evidence of successful forgetting when retained F1 has collapsed."* So judge forgetting by **how close the method is to `retrained`**, not by how low forgotten-domain accuracy is:

- `method_forgotten_prediction_agreement_with_retrained` — fraction of predictions on forgotten-domain data matching the retrained model (higher = better forgetting)
- `method_forgotten_js_divergence_from_retrained` — divergence of output probabilities from retrained (lower = better)
- `method_forgotten_linear_cka_to_retrained` — internal representation similarity to retrained (higher = better)
- `method_forgotten_abs_f1_gap_from_retrained` — |F1 gap| on forgotten domain vs retrained (lower = better)
- `method_mia_auc` vs `retrained_mia_auc`, and `method_abs_mia_auc_gap_from_retrained` — membership-inference attack AUC gap (lower = less residual privacy leakage, but only meaningful together with the utility columns)

Finer-grained versions of these live in **[all_gold_similarity.csv](4methods/all_gold_similarity.csv)** (per evaluation_domain/scope) and **[all_membership_inference.csv](4methods/all_membership_inference.csv)** (full MIA stats with bootstrap CIs).

### 2. How well the model still works on other data (retained utility)
- `method_retained_macro_f1` and `method_retained_macro_roc_auc` (compare to `original_retained_macro_f1`/`retrained_retained_macro_f1`) — overall accuracy on the domains *not* forgotten
- `method_retained_validation_log_loss` vs `original_...`/`retrained_...`

Per-domain detail (rather than macro-averaged) is in **[all_classification_metrics.csv](4methods/all_classification_metrics.csv)**: filter `evaluation_scope == "retained"` for each individual retained domain, or `evaluation_scope == "retained_macro"` for the aggregate; `model` column lets you compare `original` vs `retrained` vs each unlearning method (`ssd_canonical`, `dc_ssd`, `scrub_r`, `rollback_repair`, etc.).

### Supporting files (not the core two answers, but useful context)
- **[all_method_diagnostics.csv](4methods/all_method_diagnostics.csv)** — sanity checks so a good average doesn't hide a no-op (`is_identity_map`, `is_near_identity`, `selected_parameter_fraction`, `parameter_change_l2`)
- **[all_efficiency.csv](4methods/all_efficiency.csv)** — cost/speed of each method vs full retraining, not accuracy/privacy

**Practical tip:** load `summary.csv`, filter to the `method`/`forgotten_domain`/`architecture` you care about, and compare the `method_*` columns against both `original_*` (did it change anything?) and `retrained_*` (did it match the ideal?) — that pair of comparisons is exactly what cell 28 in the notebook tells you to read together.