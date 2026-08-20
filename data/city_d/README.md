# City D inputs (not redistributed)

This folder is intentionally empty in the public repository.

The City D district network model was provided by the operating utility and,
together with the audited leak-to-node mapping, is available from the
corresponding authors on reasonable request; the raw 2025 repair register is
restricted by the utility (see the paper's Data availability statement).

With the model and mapping in place, rebuild every derived input with:

```bash
python data/city_d_extract_corpus.py
python data/city_d_setup.py
python data/city_d_build_library.py     # resumable; re-run until COMPLETE
python data/city_d_anchor_corpus.py
```

All City D results reported in the paper are committed under `artifacts/`
(`results_city_d*.json`), so the figures and tables reproduce without this
folder; only a from-scratch re-run of the City D legs needs it. In the
committed results, DMA names are replaced by stable codes and register
leak-type labels are translated; no numeric value is altered.
