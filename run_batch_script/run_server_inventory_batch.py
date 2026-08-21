r"""Select rows from the server inventory spreadsheet for given Region x Antibody combinations.

The slides live on chead, so every run opens one SSH connection (a single password
prompt), downloads each selected slide, runs NEUSEG on it, saves the npz under
<Region>/<Antibody>/ of --output-dir, and deletes the download before moving on.
Use --dry-run to see what is on the server without processing anything.

Usage
-----
    conda activate neuseg

    python run_server_inventory_batch.py \
        --csv {path to server_inventory_ftld.csv} \
        --region MFC ANG \
        --antibody AT8 TDP43 \
        --chead-userid {chead username} \
        --n-cores {number of cores} \
        --output-dir {Path to save npz outputs} \\

--region and --antibody take several values, and every combination of the two is
selected: `--region MFC ANG --antibody AT8 TDP43` is four combinations.  Add
--dry-run to the same command to check the server without downloading anything.

Outputs go to <output-dir>/<Region>/<Antibody>/<slide>_neuseg.npz, with a log in
<output-dir>/batch_log_<regions>_<antibodies>.txt.  Slides that already have an
npz are skipped, so re-running the same command resumes an interrupted batch.
"""

import argparse
import contextlib
import os
import shlex
import subprocess
import sys
import tempfile
import time

import pandas as pd

REGIONS = ["ANG", "MFC", "OFC", "SMTC", "aCING"]
ANTIBODIES = ["AT8", "CD68", "GFAP", "HLADR", "IBA1", "NeuN", "SMI32", "TDP43"]
CHEAD_HOST = "chead.uphs.upenn.edu"

# Passed to ssh with -o so the script behaves the same for every user,
# whether or not they have a ~/.ssh/config.
SSH_OPTIONS = {
    "ControlMaster": "auto",      # reuse one connection for every ssh call
    "ServerAliveInterval": "60",  # ping every 60s so an idle VPN stays up
    "ServerAliveCountMax": "10",  # tolerate ~10 min of silence before giving up
    "TCPKeepAlive": "yes",        # keep firewall/NAT state alive
}

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 250)


def _argument_builder():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv",
                        # sibling of this script in the repo, wherever it was cloned
                        default=os.path.normpath(
                            os.path.join(os.path.dirname(__file__), "server_inventory_ftld.csv")),
                        help="path to the server inventory spreadsheet "
                             "(default: server_inventory_ftld.csv next to this script)")
    parser.add_argument("--region", nargs="+", default=REGIONS, help="regions to select")
    parser.add_argument("--antibody", nargs="+", default=ANTIBODIES, help="antibodies to select")
    parser.add_argument("--has-slide", default="True", choices=["True", "False"],
                        help="keep only rows with this HasSlide value")
    parser.add_argument("--chead-userid", required=True, help="chead username")
    parser.add_argument("--dry-run", action="store_true",
                        help="stop after reporting what is on chead")
    
    parser.add_argument("--output-dir", default=".",
                        help="save npz outputs here, under <Region>/<Antibody>/ "
                             "(default: current directory)")
    parser.add_argument("--tmp-dir",
                        default=os.path.join(tempfile.gettempdir(), "neuseg_slides"),
                        help="where each slide is downloaded to, then deleted")
    
    parser.add_argument("--main-py",
                        # main.py location in this git repo, relative to this script; the user can override it with an absolute path
                        default=os.path.normpath(
                            os.path.join(os.path.dirname(__file__), "..", "neuseg", "main.py")),
                        help="path to NEUSEG main.py (default: ../neuseg/main.py)")
    parser.add_argument("--n-cores", type=int, default=8, help="cores for main.py")
    parser.add_argument("--entrypoint", default="cells",
                        choices=["cells", "features", "tissue", "cortex"],
                        help="where main.py starts in the NEUSEG algorithm")
    parser.add_argument("--debug-level", default="normal",
                        choices=["quiet", "normal", "debug", "full"],
                        help="how much main.py prints")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        parser.error(f"no spreadsheet at {args.csv} -- pass --csv")
    if not os.path.exists(args.main_py):
        parser.error(f"no main.py at {args.main_py} -- pass --main-py")
    return args


@contextlib.contextmanager
def chead_ssh(userid):
    """Open one SSH connection to chead, reused until this block exits.

    ssh asks for the password once, on the terminal; the password never passes
    through Python.  Yields the flags that make later ssh/rsync calls share it.
    """
    # The target is the user@host string for SSH.
    target = f"{userid}@{CHEAD_HOST}"

    # ControlPath is per-process, so two runs never share a socket.
    socket = os.path.join(tempfile.gettempdir(), f"chead-ssh-{os.getpid()}")
    options = {"ControlPath": socket, **SSH_OPTIONS}
    flags = [arg for key, value in options.items() for arg in ("-o", f"{key}={value}")]

    print(f"Connecting to {target} ...")
    # Run the ssh connection in the background and check that it succeeded.  If it fails, raise an exception.
    if subprocess.run(["ssh", "-fN", *flags, target]).returncode != 0: 
        raise SystemExit(f"Could not open an SSH connection to {target}")
    try:
        # Yield the flags and target for later SSH calls to reuse this connection.
        yield flags, target
    finally:
        # Close the SSH connection when the context exits.
        subprocess.run(["ssh", "-O", "exit", *flags, target],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def check_slides(paths, flags, target):
    """Return {path: size_in_bytes} for the paths that exist on chead."""
    remote = subprocess.run(
        ["ssh", *flags, target, "tr '\\n' '\\0' | xargs -0 -r stat -c '%s %n' 2>/dev/null"],
        input="\n".join(paths), capture_output=True, text=True)
    return dict(
        (path, int(size))
        for size, path in (line.split(" ", 1) for line in remote.stdout.splitlines())
    )


def log_path(args):
    """Log named after the selection, e.g. batch_log_MFC_AT8.txt."""
    selection = f"{'-'.join(args.region)}_{'-'.join(args.antibody)}"
    return os.path.join(args.output_dir, f"batch_log_{selection}.txt")


def log(args, text=""):
    """Print a line and append it to the run log in --output-dir.

    Only whole lines go through here, never progress bars, so the log stays
    readable.  Reopening per line keeps it flushed: `tail -f` shows each slide
    the moment it finishes.
    """
    print(text)
    with open(log_path(args), "a") as handle:
        handle.write(text + "\n")


def _hms(seconds):
    """Format a duration the short way: '48m', or '3h51m' once past an hour."""
    minutes = int(seconds // 60)
    return f"{minutes // 60}h{minutes % 60:02d}m" if minutes >= 60 else f"{minutes}m"


def run_batch(df, args, flags, target):
    """Download, process, and delete one slide at a time.  Returns a tally."""
    tally = {"success": 0, "failed": 0, "skipped": 0}
    os.makedirs(args.tmp_dir, exist_ok=True)
    start = time.monotonic()
    done = 0  # slides actually processed; skipped ones are instant and would skew the ETA

    # For each row, download the slide, run NEUSEG, and delete the slide.  If any step fails, keep going to the next row.
    for n, row in enumerate(df.itertuples(), start=1):
        # Path ot .svs on chead, e.g. /data/chead/ANG/AT8/ANG_AT8_0001.svs
        slide = os.path.basename(row.ServerDirectory)
        
        # The npz output goes under <output_dir>/<Region>/<Antibody>/<slide>.npz
        out_dir = os.path.join(args.output_dir, row.Region, row.Antibody)
        npz = os.path.join(out_dir, os.path.splitext(slide)[0] + ".npz")

        # skip if the npz already exists, meaning this slide has already been processed
        if os.path.exists(npz):
            tally["skipped"] += 1
            continue

        # Rate comes from the slides processed so far, so there is no ETA on the first one.
        elapsed = time.monotonic() - start
        eta = f", ~{_hms(elapsed / done * (len(df) - n + 1))} left" if done else ""
        print(f"\n[{n}/{len(df)}] {slide}  (elapsed {_hms(elapsed)}{eta})")
        os.makedirs(out_dir, exist_ok=True)
        local_slide = os.path.join(args.tmp_dir, slide)
        slide_start, status = time.monotonic(), "OK"
        try:
            # 1. download over the SSH connection already open.  scp hands the
            # remote path to a shell on chead, so quote it: a space or a "(2)" in
            # the filename would otherwise be split into separate arguments.
            subprocess.run(["scp", *flags, f"{target}:{shlex.quote(row.ServerDirectory)}",
                            local_slide], check=True)
            # 2-3. run NEUSEG, writing the npz straight into <Region>/<Antibody>/
            subprocess.run([sys.executable, args.main_py, "-i", local_slide, "-o", out_dir,
                            "--n_cores", str(args.n_cores),
                            "--entrypoint", args.entrypoint,
                            "--debug_level", args.debug_level], check=True)
            tally["success"] += 1
        except subprocess.CalledProcessError as error:
            # Which of the two steps died, kept short so the log stays in columns.
            status = "FAIL (download)" if error.cmd[0] == "scp" else "FAIL (neuseg)"
            tally["failed"] += 1
        finally:
            # 4. always free the disk, even if the slide failed
            if os.path.exists(local_slide):
                os.remove(local_slide)
        done += 1

        # One complete record per slide, written only now that it is finished.
        log(args, f"[{n}/{len(df)}] {status:<15} {slide}  "
                  f"(took {_hms(time.monotonic() - slide_start)}, "
                  f"elapsed {_hms(time.monotonic() - start)})")

    return tally


def main():
    # [1] Parse command line arguments
    args = _argument_builder()

    # [2] Read the server inventory spreadsheet and filter rows by Region and Antibody
    df = pd.read_csv(args.csv)
    df = df[df["Region"].isin(args.region) & df["Antibody"].isin(args.antibody)]
    df = df[df["HasSlide"] == (args.has_slide == "True")]

    # The log lives in --output-dir, so that has to exist before anything is logged.
    os.makedirs(args.output_dir, exist_ok=True)

    log(args, f"\n=== run started {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    log(args, f"Regions:    {args.region}")
    log(args, f"Antibodies: {args.antibody}")
    log(args, f"HasSlide:   {args.has_slide}")
    log(args, f"Selected {len(df)} rows")

    # [3] Batch process the slides.
    # The slides to work on; they all live on chead.
    paths = sorted(df["ServerDirectory"].dropna().unique())

    # One connection for the whole run: one password prompt, no re-handshake per slide.
    with chead_ssh(args.chead_userid) as (flags, target):
        # Check which of the selected slides actually exist on chead, and how big they are.
        found = check_slides(paths, flags, target)
        log(args, f"\nOn chead: {len(found)} of {len(paths)} slides found, "
                  f"{sum(found.values()) / 1e9:.1f} GB total")

        # If --dry-run, stop here and report what is on chead.
        if args.dry_run:
            return df

        # The settings passed through to main.py for every slide.
        log(args, f"\nNEUSEG settings:")
        log(args, f"  output_dir:  {os.path.abspath(args.output_dir)}/<Region>/<Antibody>")
        log(args, f"  n_cores:     {args.n_cores}")
        log(args, f"  entrypoint:  {args.entrypoint}")
        log(args, f"  debug_level: {args.debug_level}")

        # Run NEUSEG on each slide, one at a time, downloading and deleting each slide in turn.
        tally = run_batch(df[df["ServerDirectory"].isin(found)], args, flags, target)
        log(args, f"\nDone at {time.strftime('%Y-%m-%d %H:%M:%S')}: "
                  f"{tally['success']} succeeded, {tally['failed']} failed, "
                  f"{tally['skipped']} skipped (already had an npz)")

    return df


if __name__ == "__main__":
    main()
