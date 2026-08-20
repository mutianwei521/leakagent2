# City H inputs (not redistributed)

This folder is intentionally empty in the public repository.

The City H municipal network model is utility-derived and is available from
the corresponding authors on reasonable request (see the paper's Data
availability statement). With the model in place, rebuild every derived input
of this folder with:

```bash
python data/city_h_setup.py
python data/city_h_build_library.py     # resumable; re-run until COMPLETE
python data/city_h_anchor_corpus.py
```

All City H results reported in the paper are committed under `artifacts/`
(`results_city_h*.json`), so the figures and tables reproduce without this
folder; only a from-scratch re-run of the City H leg needs it.
