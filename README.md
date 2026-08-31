# pauli_web_y_state_injection

Pauli webs of the |Y⟩ state surface code injection.

Companion code to *Pauli web of the |Y⟩ state surface code injection*
(arXiv:2501.15566).

Closed Pauli webs of the Li / Lao–Criger injection protocol are computed as the null
space of a linear system over GF(2). Nothing is asserted: the spider rule the solver
encodes is checked against the spider tensors themselves.

## What it derives

| result | |
|---|---|
| detector space after $R$ rounds | $\dim = 10 + 24(R-1)$ = $24(R-1)$ full cells + 10 half cells |
| after 1 round | only **10 of 24** checks are usable; the other 14 give uniformly random outcomes |
| after 2 rounds | **all 24**, via the round-1 × round-2 comparison |
| the paper's `+1` post-selection set | *exactly* the 10 round-1 detectors — forced, not conventional |
| initialisation errors | coverage saturates at round 1; extra rounds never add any |
| blind spots | data qubits **4** (the injected \|Y⟩) and **9** — neither error type is ever visible |
| half cells | the count **never grows**: every detector anchored on the initial states already exists after round 1 |

Two rounds is the minimum at which the code checks all of its stabilisers. Post-selection
covers the one thing no number of rounds can reach. Neither choice is derived in
Li or Lao–Criger; both fall out of the web computation.

## Files

| file | purpose |
|---|---|
| `injection_webs.py` | the diagram, the GF(2) web solver, the audit, the spider rule checked against tensors |
| `viewer.py` | writes self-contained multi-scene 3D HTML |
| `make_viewers.py` | generates `injection_webs_viewer.html` |
| `nb_two_rounds.ipynb` | the worked derivation, executed |
| `injection_webs_viewer.html` | interactive: 24 toggles on the 1-round page, 34 on the 2-round page (24 full cells + 10 init-anchored half cells = the whole detector space) |
| `fig_two_round_detector.png` | static render of the two-round detector cell |

## Running it

Needs `pyzx >= 0.10` (developed against **0.10.5**, Python 3.12.11, numpy 2.4.6 — the
notebook prints the versions it ran under).

```bash
pip install "pyzx>=0.10" numpy matplotlib jupyter
jupyter lab nb_two_rounds.ipynb
```

The 3D viewer JavaScript comes from **<https://github.com/kh428/pyzx_3d_viewer>**
(Apache-2.0), in either of two modes:

* **`js='inline'`** (default for the standalone HTML) — a local copy is embedded, so the
  page is self-contained. Point `viewer.py` at your clone with
  `export ZX_VIEWER_JS=/path/to/zx_viewer_3D.js`.
* **`js='cdn'`** (default for `inline_view` in the notebook) — imported straight from the
  repo over jsDelivr, so **nothing has to exist locally**. `viewer.CDN_REF` is pinned to a
  fixed commit of `pyzx_3d_viewer`, so the rendered pages cannot silently change under a
  future push.

If no local `zx_viewer_3D.js` is found (fresh clone), `js='inline'` falls back to the CDN
with a printed note. For a fully self-contained page, set `ZX_VIEWER_JS` to a local copy.

three.js is fetched from a CDN either way, so a live render needs the network once.
`nbconvert` stores the HTML but never runs the JavaScript, so the executed notebook looks
blank until opened in a live browser.

Note: jsDelivr serves GitHub with the correct JavaScript MIME type;
`raw.githubusercontent.com` does **not** (`text/plain` + `nosniff`), so a module script
from there is refused by the browser.

Note: the web viewer has no purple-box rendering. These scenes are plain spiders plus
Pauli webs, which it does support.

## Conventions

* green web = Z-type decoration, red web = X-type decoration
* green (Z) ancilla = X-type check, measured in X
* red (X) ancilla = Z-type check, measured in Z
* measurement outcomes are **classical** — the measurement effect is a one-legged spider
  of the ancilla's own colour, so it fuses away and no measurement leg is drawn. A check
  enters a web exactly when the web covers its ancilla **on every leg**.


## Circuit-level noise and parametric analysis (paper v5)

Code for sections 8-10 and appendices B-D of the paper.

- `circuit_noise/` - section 8. `circuits.py` builds the injection
  protocols as stim circuits with deterministic fault insertion;
  `census.py`/`dem_census.py` run the decoded-convention fault
  enumeration (table 1); `sched_search*.py` and
  `step1`-`step6_*_decoded.py` run the decoded-convention pipeline:
  per-scheme counts, the optimised schedules, the figure 22 exports
  (`step3_decoded_marks.py`), the 576-pair and heterogeneous
  restart searches, and the appendix A flat figure; `mc_*.py` sample the four schemes
  (figure 30 data), `plot_mc_v3.py` draws figure 30.
- `parametric/` - sections 9-10. `exact_ler.py` evaluates the exact
  post-selected probabilities by the parity expansion;
  `verify_25over9*.py` do the fault-pair enumeration (equation 6);
  `verify_full_*.py` the multivariate budgets (equations 7-8);
  `gen_exact_curve.py` figure 34; `y_dindep_fixed.py` /
  `t_dindep_assert.py` assert equations (3)/(5) against the
  contraction at every odd distance 3-29; `cone_check.py` and
  `interior_equality.py` back figure 33; `t_experiment.py` /
  `t_run_final.py` / `plot_t_v2.py` the |T> campaign (figure 36);
  `exact_T_cosets.py` / `exact_T_figure.py` the exact |T> curve (figure 37,
  equation 9); `gen_tsim_stages*.py` export figures 31-32 and 39.
- `figures_3d/` - the spacetime diagram builders and 3D/tikz
  exporters shared by the paper figures (schemes, patterns, webs,
  paper3d) plus per-figure `gen_*.py` scripts.
- `other_schemes/` - appendices B-D: the ZZ / hook / unitary-encoder
  schemes at distance 5 and their web verification, plus the Li
  per-gate attribution and the Lao-Criger circuit reconstruction of
  section 8.2, and the interactive viewer.
- `data/` - sampled results backing figures 30 and 36
  (`mc_results.json`, `t_results_final.json`), the t-frame exports,
  and the exact-budget output records.

Install: `pip install -r requirements.txt`.

## Licence

Apache-2.0 (see `LICENSE`), matching `pyzx` and `pyzx_3d_viewer`. Generated HTML pages
carry an attribution notice for the embedded/imported viewer JavaScript.
