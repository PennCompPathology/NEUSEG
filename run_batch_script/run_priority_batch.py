#!/usr/bin/env python3
"""
Run NEUSEG over a hand-curated priority list of slides, ahead of the full inventory.

run_server_inventory_batch.py walks server_inventory_v2.csv, which is a
subject x region x antibody x hemisphere grid.  A priority list is not a grid: it is
a flat list of slides somebody picked, and it can name regions the inventory grid has
no column for.  allison_tdp_c_slides_NU_only.csv is one such list -- 34 of its slides
are in region V1 and 2 in aITC, neither of which the inventory covers, so those 36
are unreachable through the inventory script at any filter setting.

This reads the priority CSV directly, so every row in it runs regardless of what the
inventory knows.  Everything after the slide list is the inventory script's own code,
imported rather than copied: the single SSH connection, the chead existence check, the
download/process/delete loop, and the npz completeness rule.  Stain handling comes from
the same SUBTRACT_DAB constant, so this script and the inventory script cannot drift
apart on HEM vs HEM - DAB.

Usage
-----
    ./run_priority_batch.py --chead-userid <user> --output-dir <dir> --dry-run
    ./run_priority_batch.py --chead-userid <user> --output-dir <dir>

Outputs land in <output-dir>/<Region>/<Antibody>/<slide>.npz, the same layout the
inventory script writes, with a log beside them named after the priority CSV.  Point
both scripts at the same --output-dir and the inventory run will skip whatever this
one already finished: a slide is skipped only when its npz holds every key a complete
run writes, so a partial result is redone rather than mistaken for a finished one.

The priority CSV needs these columns:
    SlidePath           absolute path to the .svs on chead
    Region, Antibody    used for the output directory
    OriginalSlideName   reported only; SlidePath's basename is what gets used
"""

import argparse
import os
import sys
import tempfile

import pandas as pd

# The inventory script is a sibling file, and importing it only defines things -- its
# main() is behind an "if __name__" guard.  sys.path is extended so this works when
# called by absolute path from another directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_server_inventory_batch as inventory


PRIORITY_CSV = os.path.join("inventory_spreadsheets", "allison_tdp_c_slides_NU_only.csv")

# Columns this script needs from the priority CSV, and what run_batch calls them.
REQUIRED_COLUMNS = ("SlidePath", "Region", "Antibody")


def parse_args():
    """Flags mirror the inventory script's, minus the grid filters it has no use for.

    --neuseg and --tmp-dir repeat that script's defaults rather than importing them,
    because _argument_builder() parses as it builds and cannot be reused piecemeal.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv",
                        default=os.path.join(os.path.dirname(__file__), PRIORITY_CSV),
                        help=f"the priority list (default: {PRIORITY_CSV} next to this script)")
    parser.add_argument("--chead-userid", required=True, help="chead username")
    parser.add_argument("--dry-run", action="store_true",
                        help="stop after reporting what is on chead and what is already done")
    parser.add_argument("--output-dir", default=".",
                        help="save npz outputs here, under <Region>/<Antibody>/ "
                             "(default: current directory)")
    parser.add_argument("--tmp-dir",
                        default=os.path.join(tempfile.gettempdir(), "neuseg_slides"),
                        help="where each slide is downloaded to, then deleted")
    parser.add_argument("--neuseg",
                        default=os.path.join(os.path.dirname(sys.executable), "neuseg"),
                        help="path to the neuseg command (default: alongside this python)")
    parser.add_argument("--n-cores", type=int, default=8, help="cores for main.py")
    parser.add_argument("--entrypoint", default="cells",
                        choices=["cells", "features", "wm", "cortex"],
                        help="where main.py starts in the NEUSEG algorithm")
    parser.add_argument("--staining-code", default="HDAB", choices=["HDAB", "CVDAB"],
                        help="how the slides were stained")
    parser.add_argument("--debug-level", default="normal",
                        choices=["quiet", "normal", "debug", "full"],
                        help="how much main.py prints")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        parser.error(f"no priority list at {args.csv} -- pass --csv")
    if not os.path.exists(args.neuseg):
        parser.error(f"no neuseg command at {args.neuseg} -- run "
                     "`pip install -e . --no-deps` in the NEUSEG repo, or pass --neuseg")
    return args


def load_priority(csv_path):
    """The priority CSV as a frame run_batch can iterate.

    run_batch reads row.ServerDirectory for the download and row.Region / row.Antibody
    for the output directory, so SlidePath is renamed to match rather than teaching
    run_batch a second column name.
    """
    df = pd.read_csv(csv_path, dtype=str)

    absent = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if absent:
        raise SystemExit(f"{csv_path} is missing required column(s): {', '.join(absent)}")

    df = df.rename(columns={"SlidePath": "ServerDirectory"})

    # A row with no path or no output directory cannot be acted on; dropping them here
    # keeps the counts below honest instead of failing one slide at a time.
    before = len(df)
    df = df.dropna(subset=["ServerDirectory", "Region", "Antibody"])
    if len(df) < before:
        print(f"  dropped {before - len(df)} row(s) with a missing path, Region, or Antibody")

    # The same slide listed twice would be downloaded twice and produce the same npz.
    before = len(df)
    df = df.drop_duplicates(subset=["ServerDirectory"])
    if len(df) < before:
        print(f"  dropped {before - len(df)} duplicate path(s)")

    return df.reset_index(drop=True)


def describe(df, args):
    """What the list contains, and how much of it is already done."""
    print("=== priority list ===")
    print(f"  file:     {args.csv}")
    print(f"  slides:   {len(df)}")
    if "AutopsyID" in df.columns:
        print(f"  subjects: {df['AutopsyID'].nunique()}")
    # int() on the counts: numpy scalars print as "np.int64(34)" inside a dict repr.
    for col in ("Region", "Antibody"):
        counts = {k: int(v) for k, v in df[col].value_counts().items()}
        print(f"  {col.lower() + ':':<10}{counts}")

    # Same completeness rule run_batch applies, reported up front so a re-run says how
    # much work is actually left rather than discovering it slide by slide.
    states = {"complete": 0, "incomplete": 0, "corrupt": 0, "missing": 0}
    for row in df.itertuples():
        npz = os.path.join(args.output_dir, row.Region, row.Antibody,
                           os.path.splitext(os.path.basename(row.ServerDirectory))[0] + ".npz")
        state = inventory.npz_state(npz)
        states["incomplete" if state.startswith("incomplete") else state] += 1

    print("\n=== already in the output directory ===")
    print(f"  {os.path.abspath(args.output_dir)}/<Region>/<Antibody>")
    for key in ("complete", "incomplete", "corrupt", "missing"):
        print(f"  {key:<11}{states[key]}")
    print(f"\n  {len(df) - states['complete']} slide(s) still to run")
    return states


def main():
    args = parse_args()

    df = load_priority(args.csv)
    describe(df, args)

    # The inventory script names its log after the grid filters it was given, which a
    # flat list has none of.  Naming it after the CSV keeps priority runs from writing
    # into an inventory run's log when both share an --output-dir.
    tag = os.path.splitext(os.path.basename(args.csv))[0]
    inventory.log_path = lambda a: os.path.join(a.output_dir, f"batch_log_priority_{tag}.txt")

    os.makedirs(args.output_dir, exist_ok=True)

    paths = sorted(df["ServerDirectory"].unique())
    with inventory.chead_ssh(args.chead_userid) as (flags, target):
        found = inventory.check_slides(paths, flags, target)
        print(f"\n=== chead ===")
        print(f"  {len(found)} of {len(paths)} slides found"
              f"  ({sum(found.values()) / 1e9:.1f} GB)")

        absent = [p for p in paths if p not in found]
        if absent:
            print(f"  {len(absent)} NOT on chead -- these cannot be run:")
            for p in absent[:10]:
                print(f"    {p}")
            if len(absent) > 10:
                print(f"    ... and {len(absent) - 10} more")

        if args.dry_run:
            print("\nDry run: nothing was downloaded or processed.")
            return

        runnable = df[df["ServerDirectory"].isin(found)]
        if runnable.empty:
            print("\nNothing to run.")
            return

        inventory.log(args, f"=== priority batch: {tag} ===")
        inventory.log(args, f"  csv:          {args.csv}")
        inventory.log(args, f"  output_dir:   {os.path.abspath(args.output_dir)}"
                            "/<Region>/<Antibody>")
        inventory.log(args, f"  slides:       {len(runnable)}")
        inventory.log(args, f"  n_cores:      {args.n_cores}")
        inventory.log(args, f"  entrypoint:   {args.entrypoint}")
        inventory.log(args, f"  staining:     {args.staining_code}")
        inventory.log(args, f"  subtract_dab: {inventory.SUBTRACT_DAB}   "
                            f"({'HEM - DAB' if inventory.SUBTRACT_DAB else 'raw HEM'})")
        inventory.log(args, f"  debug_level:  {args.debug_level}")
        inventory.log(args)

        tally = inventory.run_batch(runnable, args, flags, target)

    inventory.log(args, f"\nDone: {tally['success']} ok, {tally['failed']} failed, "
                        f"{tally['skipped']} skipped")
    print(f"\nLog: {inventory.log_path(args)}")


if __name__ == "__main__":
    main()
