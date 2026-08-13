#!/usr/bin/env python3
"""Run main.py over every slide in the cohort, preserving the group layout."""

import os
import sys
import time
import subprocess
from datetime import datetime

COHORT = '/home/hsroh/Research/bvFTD_Eval_cohort'
OUTPUT = '/home/hsroh/Research/bvFTD_Eval_cohort/NEUSEG_RESULTS_081226'
GROUPS = ('FTLD-TAU', 'FTLD-TDP')
N_CORES = 128
MAIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'main.py')

DEBUG_LEVEL = 'debug'
ENTRY_POINT = 'cortex'
SUMMARY_TEXT_FILENAME = f'run_summary_{DEBUG_LEVEL}.txt'

print('Starting Batch Running - ENTRY_POINT: {ENTRY_POINT} / DEBUG_LEVEL: {DEBUG_LEVEL}')

slides = [(group, f) for group in GROUPS
          for f in sorted(os.listdir(os.path.join(COHORT, group)))
          # skip the ._ AppleDouble sidecars macOS leaves on the external drive,
          # which double the file count and are not slides
          if f.endswith('.svs') and not f.startswith('.')]

succeeded, failed, skipped = [], [], []
for i, (group, fname) in enumerate(slides, 1):
    name = f"{group}/{fname}"
    out_dir = os.path.join(OUTPUT, group, os.path.splitext(fname)[0])

    # # gm_mask.npy is the last thing main.py writes, so its presence means the slide
    # # finished: the cohort takes hours, and this makes the run resumable
    # if os.path.exists(os.path.join(out_dir, 'gm_mask.npy')):
    #     skipped.append(name)
    #     continue

    print(f"[{i}/{len(slides)}] {name} ... ", end='', flush=True)
    t0 = time.time()
    # capture_output keeps main.py's logging and progress bars off the terminal;
    # they are written beside that slide's results instead
    # result = subprocess.run([sys.executable, MAIN,
    #                          '-i', os.path.join(COHORT, group, fname),
    #                          '-o', out_dir,
    #                          '--n_cores', str(N_CORES)],
    #                         capture_output=True, text=True)
    result = subprocess.run([sys.executable, MAIN,
                             '-i', os.path.join(COHORT, group, fname),
                             '-o', out_dir,
                             '--n_cores', str(N_CORES)],
                             '--entrypoint', ENTRY_POINT,
                             '--debug_level', DEBUG_LEVEL,
                            capture_output=True, text=True)

    os.makedirs(out_dir, exist_ok=True)      # main.py may have failed before making it
    with open(os.path.join(out_dir, 'neuseg_log.txt'), 'w') as f:
        f.write(result.stdout + result.stderr)

    # keep going: one bad slide should not end the batch
    ok = result.returncode == 0
    (succeeded if ok else failed).append(name)
    print(f"{'ok' if ok else 'FAILED'} ({(time.time() - t0) / 60:.1f} min)", flush=True)

# --- summary, printed and written beside the results ---
lines = [f"NEUSEG cohort run  {datetime.now():%Y-%m-%d %H:%M}",
         f"{len(succeeded)} succeeded, {len(failed)} failed, {len(skipped)} skipped "
         f"(of {len(slides)} slides)"]
for title, group in (('FAILED', failed), ('SKIPPED', skipped), ('SUCCEEDED', succeeded)):
    lines += [f"\n{title} ({len(group)})"] + [f"  {n}" for n in group]

os.makedirs(OUTPUT, exist_ok=True)
with open(os.path.join(OUTPUT, SUMMARY_TEXT_FILENAME), 'w') as f:
    f.write("\n".join(lines) + "\n")

print(f"\n{lines[1]}")
for n in failed:
    print(f"  FAILED  {n}")
print(f"summary: {os.path.join(OUTPUT, SUMMARY_TEXT_FILENAME)}")
