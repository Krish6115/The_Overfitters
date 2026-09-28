# Legacy & Experimental Pipeline Archive

This directory archives the iterative experimental scripts developed during the project lifecycle, tracking the engineering progression from initial baselines to state-of-the-art performance.

## File Index & Experimental Evolution

| Script | F0.5 Score | Key Technique / Innovation | Insight / Failure Mode |
|---|---|---|---|
| `kaggle_pipeline.py` | 80.9% | Multi-route char 3-4 n-gram TF-IDF blocking + 8 LightGBM features. | Fast and scalable, but suffered from 54,000 duplicate target claims. |
| `kaggle_pipeline_dense.py` | — | Dense semantic vector blocking using multilingual embeddings. | High computational overhead; dense cosine struggled with fine-grained address numbers compared to character n-grams. |
| `rescorer_v2.py` | 81.1% | 18 rich lexical features + greedy 1-to-1 target assignment. | Resolved duplicate collisions; established that each gallery target can belong to at most one query. |
| `educational_recovery.py` | 81.0% | Transitive entity link exploration with address token relaxation. | Over-relaxed postal matching pruned 7,100 true matches due to minor address formatting variations. |
| `target85_optimizer.py` | 81.9% | Strict dual-constraint filtering: Name Token-Set $\ge 85\%$ & Addr Token-Set $\ge 65\%$. | Recovered +177,000 valid links without polluting precision. |
| `deep_structural_optimizer.py` | 81.9% | Top-15 candidate depth scan targeting asymmetric matches ($S1 \to S2$ without $S3$). | Proved that cross-gallery recovery is safest when anchored on already validated entities. |
| `precision_max_sota.py` | **82.0%** | Ultra-strict composite verification ($\ge 95\%$ similarity, street number matching) + Strict Singleton Guard. | **Peak All-Time Performance:** Protected 100% of singletons while completing +46,000 high-confidence missing sibling links. |

## Historical Notes

- **The Singleton Vulnerability:** In Macro $F_{0.5}$, precision is weighted 4x higher than recall ($1.25 \times P \times R / (0.25 P + R)$). A single false positive on an unlinked singleton causes the score for that entity to drop instantly from $1.0 \to 0.0$. Attempting aggressive low-threshold recall recovery (e.g. 0.50 threshold) collapsed score to 76.7%.
- **Optimal Production Code:** For production usage, please use the refactored, modular package in `src/` and entrypoints in `scripts/run_pipeline.py` and `scripts/optimize_precision.py`.
