import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tsim
from t_experiment import run

out = {}
for dd, p2, shots in [(3, 2e-3, 5_000_000), (5, 2e-3, 5_000_000),
                      (5, 1e-3, 8_000_000)]:
    for state, pred in [('T', 2.5/15), ('Y', 3/15)]:
        acc, bad = run(state, p2, shots, tsim, dd=dd)
        out[f'{state}|d{dd}|{p2}'] = (acc, bad)
        print(f'{state} d={dd} p2={p2:.0e}: acc={acc}, LER/p2='
              f'{bad/acc/p2:.3f} +- {bad**0.5/acc/p2:.3f} '
              f'(predict {pred:.3f})', flush=True)
json.dump(out, open('t_results_final.json', 'w'))
print('saved')
