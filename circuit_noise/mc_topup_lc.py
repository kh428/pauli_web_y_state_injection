"""Top-up shots for the Lao-Criger schedule curves (CR-LC, MR-LC):
4x the original per-point target, capped, merged into mc_results.json
(counts are additive: accepted/rejected/logical-error tallies)."""
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from concurrent.futures import ProcessPoolExecutor
from mc_par import one

def main():
    tasks = []
    for d, cap in [(3, 1_200_000_000), (5, 900_000_000), (7, 600_000_000),
                   (9, 400_000_000), (11, 250_000_000)]:
        for name in ('CR-LC', 'MR-LC'):
            for model, ps in [('p2', [2e-4, 5e-4, 1e-3, 2e-3]),
                              ('all', [3e-4, 1e-3])]:
                for p in ps:
                    shots = min(int(300_000 / (0.4 * p)), cap)
                    tasks.append((d, name, model, p, shots))
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '..', 'mc_results.json')
    res = json.load(open(path))
    done = 0
    with ProcessPoolExecutor(max_workers=10) as ex:
        for key, acc, rej, err in ex.map(one, tasks):
            if key in res:
                a0, r0, e0 = res[key]
                res[key] = [a0 + acc, r0 + rej, e0 + err]
            else:
                res[key] = [acc, rej, err]
            done += 1
            json.dump(res, open(path, 'w'))
            print(f'{done}/{len(tasks)} {key} merged', flush=True)
    print('topup complete')

if __name__ == '__main__':
    main()
