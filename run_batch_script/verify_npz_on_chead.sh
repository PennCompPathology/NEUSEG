#!/usr/bin/env bash
#
# Verify that every .npz and *_log.pkl under the local results directory is
# also on chead, before anything is deleted here.
#
# copy_npz_to_chead.sh counts files when it finishes; counting is not
# comparing.  rsync -rlt calls a file "already there" when its size and
# timestamp match, so neither the copy logs nor a file tally can tell you the
# bytes on chead are the bytes here.  This does the comparison properly.
#
# Two levels, cheap first:
#
#   size   (default)    one remote find, compared against the local tree.
#                       Catches anything missing, truncated, or a different
#                       length.  Minutes, and no data leaves either machine.
#   --checksum          rsync --dry-run --checksum: reads every byte on both
#                       sides and compares digests.  Catches silent corruption
#                       that kept its length.  Hours for ~440GB, but it
#                       transfers nothing.
#
# Nothing is ever written or deleted, on either machine.  The exit status is
# the answer: 0 = every local file is on chead and matches, 1 = it is not safe
# to delete, 2 = the check could not be run.
#
# Usage
# -----
#     ./verify_npz_on_chead.sh --chead-userid <user>              # size check
#     ./verify_npz_on_chead.sh --chead-userid <user> --checksum   # byte check
#
# Mismatches are written to a report file, one path per line, so a re-copy can
# be driven straight from it with rsync --files-from.

set -euo pipefail

CHEAD_HOST="chead.uphs.upenn.edu"
SRC="/home/hsroh/Research/bvFTD_Eval_cohort/chead_batch_results"
DEST="/data/hippogang/histology/PDNL/neuseg/v1_0"
USERID=""
MODE="size"
INCLUDE_PKL=1
REPORT=""

usage() {
    awk 'NR > 2 && /^#/ { sub(/^# ?/, ""); print; next } NR > 2 { exit }' "$0"
    cat <<'USAGE_EOF'

Options
  --chead-userid USER   chead username (required)
  --src DIR             local results directory
  --dest DIR            the copy on chead
  --checksum            compare file contents, not just sizes (slow)
  --no-pkl              check only the npz, ignoring the *_log.pkl companions
  --report FILE         write mismatching paths here (default: alongside --src)
  -h, --help            this message
USAGE_EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --chead-userid) USERID="${2:?--chead-userid needs a value}"; shift 2 ;;
        --src)          SRC="${2:?--src needs a value}";             shift 2 ;;
        --dest)         DEST="${2:?--dest needs a value}";           shift 2 ;;
        --report)       REPORT="${2:?--report needs a value}";       shift 2 ;;
        --checksum)     MODE="checksum"; shift ;;
        --no-pkl)       INCLUDE_PKL=0;   shift ;;
        -h|--help)      usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; echo >&2; usage >&2; exit 2 ;;
    esac
done

[[ -n "$USERID" ]] || { echo "error: --chead-userid is required" >&2; exit 2; }
[[ -d "$SRC" ]]    || { echo "error: no such source directory: $SRC" >&2; exit 2; }

SRC="${SRC%/}"
DEST="${DEST%/}"
TARGET="$USERID@$CHEAD_HOST"
REPORT="${REPORT:-$SRC/verify_against_chead_$(date +%Y%m%d_%H%M%S).txt}"

# The same name filter copy_npz_to_chead.sh sends, so the two agree on what
# "every file" means.  Anything the copy never sent must not be demanded here.
if [[ $INCLUDE_PKL -eq 1 ]]; then
    find_expr=(-name '*.npz' -o -name '*_log.pkl')
else
    find_expr=(-name '*.npz')
fi

# ------------------------------------------------------------- one connection
SOCKET="${TMPDIR:-/tmp}/chead-verify-$$"
SSH_FLAGS=(-o "ControlPath=$SOCKET" -o ControlMaster=auto
           -o ServerAliveInterval=60 -o ServerAliveCountMax=10 -o TCPKeepAlive=yes)

echo "Connecting to $TARGET ..."
ssh -fN "${SSH_FLAGS[@]}" "$TARGET" \
    || { echo "error: could not open an SSH connection to $TARGET" >&2; exit 2; }
work="$(mktemp -d)"
cleanup() {
    rm -rf "$work"
    ssh -O exit "${SSH_FLAGS[@]}" "$TARGET" >/dev/null 2>&1
    return 0
}
trap cleanup EXIT

ssh "${SSH_FLAGS[@]}" "$TARGET" "[ -d $(printf '%q' "$DEST") ]" \
    || { echo "error: $DEST does not exist on chead" >&2; exit 2; }

# --------------------------------------------------------------- the size pass
# "<size>\t<path-relative-to-root>" from each side, sorted, then joined on the
# path.  Relative paths are what make the two sides comparable at all.
#
# Tab-separated, not space-separated: 516 of these slide names contain a space,
# so splitting "<size> <path>" on whitespace would truncate the path at the
# first one and silently compare the wrong files.  No name here holds a tab or
# a newline, which is what makes a tab safe as the delimiter.
echo "Listing local files under $SRC ..."
find "$SRC" -type f \( "${find_expr[@]}" \) -printf '%s\t%P\n' \
    | LC_ALL=C sort -t$'\t' -k2,2 > "$work/local"

echo "Listing remote files under $DEST ..."
remote_find="find $(printf '%q' "$DEST") -type f \\( $(
    [[ $INCLUDE_PKL -eq 1 ]] && echo "-name '*.npz' -o -name '*_log.pkl'" \
                             || echo "-name '*.npz'" ) \\) -printf '%s\\t%P\\n'"
ssh "${SSH_FLAGS[@]}" "$TARGET" "$remote_find" \
    | LC_ALL=C sort -t$'\t' -k2,2 > "$work/remote"

local_n=$(wc -l < "$work/local")
remote_n=$(wc -l < "$work/remote")
local_b=$(awk -F'\t' '{s+=$1} END {print s+0}' "$work/local")
remote_b=$(awk -F'\t' '{s+=$1} END {print s+0}' "$work/remote")

echo
echo "=== inventory ==="
printf "  here:      %7d files  %s bytes\n" "$local_n" "$local_b"
printf "  on chead:  %7d files  %s bytes\n" "$remote_n" "$remote_b"
echo

# Join on the path field.  Three outcomes per local file: absent on chead,
# present with a different size, or present and the same length.
awk -F'\t' -v OFS='\t' 'NR == FNR { size[$2] = $1; next }
     { if (!($2 in size))        print "ABSENT", $2
       else if (size[$2] != $1)  print "SIZE", $2, "here=" $1, "chead=" size[$2] }' \
    "$work/remote" "$work/local" > "$work/bad"

# Files on chead that nothing here accounts for.  Harmless for deleting, but
# worth seeing: it usually means the destination holds another run's output.
awk -F'\t' -v OFS='\t' 'NR == FNR { seen[$2] = 1; next } !($2 in seen) { print "EXTRA", $2 }' \
    "$work/local" "$work/remote" > "$work/extra"

absent=$(grep -c '^ABSENT' "$work/bad" || true)
sizebad=$(grep -c '^SIZE'  "$work/bad" || true)
extra=$(wc -l < "$work/extra")

echo "=== size check ==="
printf "  missing on chead:        %d\n" "$absent"
printf "  present but wrong size:  %d\n" "$sizebad"
printf "  on chead only (extra):   %d\n" "$extra"
echo

# ----------------------------------------------------------- the checksum pass
# Only worth running if the sizes already line up; otherwise fix those first.
checkbad=0
if [[ "$MODE" == "checksum" ]]; then
    if (( absent + sizebad > 0 )); then
        echo "Skipping the checksum pass: fix the size mismatches above first."
    else
        echo "=== checksum check ==="
        echo "  Reading every byte on both sides.  This takes hours; nothing is transferred."
        filters=(--include='*/' --include='*.npz')
        [[ $INCLUDE_PKL -eq 1 ]] && filters+=(--include='*_log.pkl')
        filters+=(--exclude='*' --prune-empty-dirs)
        # -i marks what differs; with --checksum and --dry-run, a line beginning
        # ">f" means the contents diverge.  No other flag reports that.
        rsync -rlt --checksum --dry-run --itemize-changes --omit-dir-times \
              -e "ssh ${SSH_FLAGS[*]}" "${filters[@]}" \
              "$SRC/" "$TARGET:$DEST/" 2>"$work/rsync.err" \
            | awk -v OFS='\t' '/^>f/ { print "CONTENT", substr($0, 13) }' \
            > "$work/content" || {
                echo "  rsync reported a problem:" >&2; cat "$work/rsync.err" >&2; exit 2; }
        checkbad=$(wc -l < "$work/content")
        printf "  contents differ:         %d\n" "$checkbad"
        echo
    fi
fi

# ----------------------------------------------------------------- the verdict
cat "$work/bad" "$work/content" "$work/extra" 2>/dev/null > "$REPORT" || true
PATHS="${REPORT%.txt}.paths"
awk -F'\t' '$1 != "EXTRA" { print $2 }' "$REPORT" > "$PATHS"
total=$(( absent + sizebad + checkbad ))

if (( total == 0 )); then
    if [[ "$MODE" == "checksum" ]]; then
        echo "VERIFIED: every local file is on chead with identical contents."
    else
        echo "VERIFIED (sizes): every local file is on chead with a matching size."
        echo "Re-run with --checksum to compare the bytes before deleting."
    fi
    (( extra > 0 )) && echo "Note: chead holds $extra file(s) not present here; see $REPORT"
    echo "Report: $REPORT"
    exit 0
fi

echo "NOT SAFE TO DELETE: $total file(s) are missing or do not match."
echo "Report: $REPORT"
echo
echo "The bare paths are in $PATHS; to send just those up:"
echo "  rsync -rlt --omit-dir-times --files-from=$(printf '%q' "$PATHS") \\"
echo "    $(printf '%q' "$SRC/") $TARGET:$(printf '%q' "$DEST/")"
exit 1
