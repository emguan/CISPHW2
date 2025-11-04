"""
Used to find mean relative error between produced files and given debug files.

Executable:
python -m tests.results_analysis.error

Output: 
[a] mean relative error per column: [5.23012e-05, 0, 0.00012989]
[b] mean relative error per column: [0.000842862, 0.00058196, 0.000423972]
[c] mean relative error per column: [0.000236889, 2.32472e-05, 0.000135929]
[d] mean relative error per column: [4.52735e-05, 0, 3.16937e-05]
[e] mean relative error per column: [0.000495959, 0.000817346, 0.00034983]
[f] mean relative error per column: [0.00206805, 0.000745248, 0.00113843]

Author: Emily Guan
"""

import numpy as np
from pathlib import Path

DEBUG_FILES = ["a", "b", "c", "d", "e", "f"]

def filehandle_actual(letter):
    return f"data/pa2-debug-{letter}-output2.txt"

def filehandle_obs(letter):
    return f"output2/pa2-debug-{letter}-EM-output2.txt"

"""
Reads files with header: 'N,name' followed by N CSV rows of numbers.
"""
def read_em_file(path: str) -> np.ndarray:
    with open(path, "r") as f:

        first = f.readline().strip()

        left, *_ = first.split(",", 1)

        n_rows = int(left.strip())

        arr = np.loadtxt(f, delimiter=",", max_rows=n_rows)

        arr = np.atleast_2d(arr)

    return arr

"""
Mean relative error per column: mean( |obs - actual| / max(|actual|, eps) )
"""
def calc_error(actual: np.ndarray, obs: np.ndarray) -> np.ndarray:
    
    actual = np.asarray(actual)
    obs = np.asarray(obs)
    
    diff = obs - actual
    denom = np.maximum(np.abs(actual), 1e-12)  # avoid zero division
    
    rel = np.abs(diff) / denom

    return np.mean(rel, axis=0) # taking average relative error

def main():
    results = {}
    for letter in DEBUG_FILES:
        a_path = filehandle_actual(letter)
        o_path = filehandle_obs(letter)

        actual = read_em_file(a_path)
        obs = read_em_file(o_path)

        err = calc_error(actual, obs)
        results[letter] = err

        cols = ", ".join(f"{e:.6g}" for e in err)
        print(f"[{letter}] mean relative error per column: [{cols}]")

if __name__ == "__main__":
    main()
