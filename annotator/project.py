#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""Project model for the SANA ROI Annotator.

A project is a folder holding a manifest and the annotations made against it,
in the spirit of a QuPath project:

    MyProject/
        project.json                              manifest: user + slide entries
        annotations/<slide>_annotations.geojson    ROIs drawn in this project

The slides themselves are never copied. The manifest records where each `.npz`
lives, so an archive can sit on an external drive and still be annotated from a
project on the local disk.

This module is deliberately free of Qt: everything here is plain file handling,
so the pairing and validation rules can be exercised without a GUI.

Layout of this file:

    1. SlideEntry - one slide, its optional log sidecar, and its status
    2. Project    - the manifest: create, open, save, add slides
"""

import os
import csv
import json
import datetime

import numpy as np

PROJECT_FILE = "project.json"
ANNOTATION_DIR = "annotations"
PROJECT_VERSION = 1

ANNOTATION_SUFFIX = "_annotations.geojson"
LOG_SUFFIX = "_log.pkl"

# quality control lives in one sheet per project; the statuses are expected to
# grow, so everything downstream reads them from this tuple
QC_SUFFIX = "_QC.csv"
QC_STATUSES = ("Minor errors", "Fail")

# bad tissue is a flag rather than a status: a slide can be unusable tissue and
# still be graded, so it gets its own column
QC_COLUMNS = ("Slidename", "QC Status", "Bad Tissue", "Comments")
QC_EMPTY = {"status": "", "bad_tissue": False, "comments": ""}

# an archive is only a slide if it carries every array the annotator renders;
# a partial pipeline run is rejected at import rather than failing on load
REQUIRED_ARRAYS = ("thumbnail", "feature_heatmap", "gm_mask", "wm_mask",
                   "tissue_mask", "gmwm_contours")

# entry states, in the order the panel colours them
STATUS_OK = "ok"            # .npz and _log.pkl both present
STATUS_NO_LOG = "no_log"    # .npz only - the annotator runs fine without the log
STATUS_MISSING = "missing"  # the .npz has gone away since it was imported


# ---------------------------------------------------------------------------
# 1. SlideEntry
# ---------------------------------------------------------------------------

class SlideEntry:
    """One slide in a project: its archive, and the log sidecar if there is one."""

    def __init__(self, npz_path, log_path=None):
        self.npz_path = os.path.abspath(npz_path)
        self.log_path = os.path.abspath(log_path) if log_path else None

    @property
    def name(self):
        """The WSI name, i.e. the archive filename without its extension."""
        return os.path.splitext(os.path.basename(self.npz_path))[0]

    @property
    def status(self):
        """Whether this entry is fully, partly, or no longer usable."""
        if not os.path.exists(self.npz_path):
            return STATUS_MISSING
        if self.log_path and os.path.exists(self.log_path):
            return STATUS_OK
        return STATUS_NO_LOG

    def to_dict(self):
        return {"name": self.name, "npz": self.npz_path, "log": self.log_path}

    @classmethod
    def from_dict(cls, data):
        return cls(data["npz"], data.get("log"))


def find_log_file(npz_path):
    """The `<slide>_log.pkl` sitting beside an archive, or None."""
    log_path = os.path.splitext(npz_path)[0] + LOG_SUFFIX
    return log_path if os.path.exists(log_path) else None


def is_slide_archive(path):
    """True when the file is an .npz holding every array the annotator needs."""
    if not path.endswith(".npz"):
        return False
    try:
        with np.load(path, allow_pickle=True) as z:
            return all(k in z.files for k in REQUIRED_ARRAYS)
    except Exception:
        return False


def common_relocation(old_path, new_path):
    """The (old_prefix, new_prefix) implied by one file having moved.

    Everything the two paths share at the end is the part that did not move,
    so what precedes it is the substitution to apply to the other entries.
    """
    old_parts, new_parts = old_path.split(os.sep), new_path.split(os.sep)

    shared = 0
    while (shared < len(old_parts) and shared < len(new_parts)
           and old_parts[-1 - shared] == new_parts[-1 - shared]):
        shared += 1

    return (os.sep.join(old_parts[:len(old_parts) - shared]),
            os.sep.join(new_parts[:len(new_parts) - shared]))


def collect_slide_paths(paths):
    """Split dropped or chosen paths into archives and log sidecars.

    Directories are searched recursively, so a whole cohort tree can be dropped
    in one go. Anything that is neither a valid archive nor a log is returned as
    rejected, with a short reason, so the caller can tell the user why.
    """
    npz_paths, log_paths, rejected = [], {}, []

    # flatten directories into the files underneath them
    files = []
    for path in paths:
        if os.path.isdir(path):
            for dirpath, dirnames, filenames in os.walk(path):
                dirnames.sort()
                files += [os.path.join(dirpath, f) for f in sorted(filenames)]
        else:
            files.append(path)

    for path in files:
        if path.endswith(LOG_SUFFIX):
            # keyed by slide name so it can be matched to its archive below
            log_paths[os.path.basename(path)[:-len(LOG_SUFFIX)]] = path
        elif path.endswith(".npz"):
            if is_slide_archive(path):
                npz_paths.append(path)
            else:
                rejected.append((path, "missing required arrays"))
        elif os.path.isfile(path):
            rejected.append((path, "not an .npz or _log.pkl"))

    return sorted(npz_paths), log_paths, rejected


# ---------------------------------------------------------------------------
# 2. Project
# ---------------------------------------------------------------------------

class Project:
    """The manifest: which user is annotating, and which slides are in scope."""

    def __init__(self, path, user="", entries=None, created=None):
        self.path = os.path.abspath(path)
        self.user = user
        self.entries = entries if entries is not None else []
        self.created = created or datetime.datetime.now().isoformat(timespec="seconds")
        self.qc = {}            # slide name -> (status, comments)

    # -- locations ----------------------------------------------------------

    @property
    def manifest_path(self):
        return os.path.join(self.path, PROJECT_FILE)

    @property
    def annotation_dir(self):
        return os.path.join(self.path, ANNOTATION_DIR)

    def annotation_path(self, entry):
        """Where this slide's ROIs live - inside the project, not next to the
        archive, so the slide drive stays untouched and the project is portable.

        The folder is recreated if missing: an empty `annotations/` does not
        survive git or most file sync, so a freshly cloned project would
        otherwise fail on the first save.
        """
        os.makedirs(self.annotation_dir, exist_ok=True)
        return os.path.join(self.annotation_dir, entry.name + ANNOTATION_SUFFIX)

    def annotation_count(self, entry):
        """How many ROIs are already saved for this slide, 0 if none.

        A GM ROI is stored as four separate boundary curves sharing one name,
        so the distinct names are counted rather than the features. An empty
        file - left behind by Reset Annotations - correctly counts as zero.
        """
        path = self.annotation_path(entry)
        if not os.path.exists(path):
            return 0
        try:
            with open(path, "r") as fp:
                features = json.load(fp)
            return len({f.get("properties", {}).get("name", "") for f in features})
        except Exception:
            return 0

    @property
    def name(self):
        return os.path.basename(self.path)

    # -- quality control ----------------------------------------------------

    @property
    def qc_path(self):
        return os.path.join(self.path, self.name + QC_SUFFIX)

    def read_qc(self):
        """Load the QC sheet into {slide name: record}.

        Missing columns read as empty, so a sheet written before a column
        existed still loads.
        """
        qc = {}
        if not os.path.exists(self.qc_path):
            return qc
        try:
            with open(self.qc_path, newline="") as fp:
                for row in csv.DictReader(fp):
                    name = (row.get(QC_COLUMNS[0]) or "").strip()
                    if name:
                        qc[name] = {
                            "status": (row.get(QC_COLUMNS[1]) or "").strip(),
                            "bad_tissue": (row.get(QC_COLUMNS[2]) or "").strip().lower()
                                          in ("yes", "true", "1"),
                            "comments": (row.get(QC_COLUMNS[3]) or "").strip()}
        except Exception:
            pass
        return qc

    def write_qc(self):
        """Rewrite the whole sheet, rows in slide-list order.

        Rewriting rather than appending is what keeps the order matching the
        panel and drops rows for slides that have been removed.
        """
        rows = [(e.name, r["status"], "Yes" if r["bad_tissue"] else "", r["comments"])
                for e in self.entries for r in [self.qc.get(e.name)] if r]
        if not rows and not os.path.exists(self.qc_path):
            return
        with open(self.qc_path, "w", newline="") as fp:
            writer = csv.writer(fp)
            writer.writerow(QC_COLUMNS)
            writer.writerows(rows)

    def set_qc(self, name, status, bad_tissue, comments):
        """Record or clear one slide's QC, then rewrite the sheet.

        The row is kept while anything is set, so a comment on its own is not
        silently thrown away.
        """
        if status or bad_tissue or comments:
            self.qc[name] = {"status": status, "bad_tissue": bad_tissue,
                             "comments": comments}
        else:
            self.qc.pop(name, None)
        self.write_qc()

    def qc_record(self, name):
        return self.qc.get(name, QC_EMPTY)

    def qc_status(self, name):
        return self.qc_record(name)["status"]

    # -- lifecycle ----------------------------------------------------------

    @classmethod
    def create(cls, path, user):
        """Start a new project folder. The user name is asked for only here."""
        project = cls(path, user=user)
        os.makedirs(project.annotation_dir, exist_ok=True)
        project.save()
        return project

    @classmethod
    def open(cls, path):
        """Load an existing project folder, by the folder or its manifest."""
        if os.path.isfile(path):
            path = os.path.dirname(path)

        manifest = os.path.join(path, PROJECT_FILE)
        if not os.path.exists(manifest):
            raise FileNotFoundError(f"No {PROJECT_FILE} in {path}")

        with open(manifest, "r") as fp:
            data = json.load(fp)

        project = cls(
            path,
            user=data.get("user", ""),
            entries=[SlideEntry.from_dict(e) for e in data.get("entries", [])],
            created=data.get("created"),
        )
        project.qc = project.read_qc()
        return project

    def save(self):
        """Rewrite the manifest. Called after every change so nothing is lost."""
        os.makedirs(self.annotation_dir, exist_ok=True)
        data = {
            "version": PROJECT_VERSION,
            "user": self.user,
            "created": self.created,
            "entries": [e.to_dict() for e in self.entries],
        }
        with open(self.manifest_path, "w") as fp:
            json.dump(data, fp, indent=2)

    @staticmethod
    def is_project(path):
        return os.path.exists(os.path.join(path, PROJECT_FILE))

    # -- contents -----------------------------------------------------------

    def add_paths(self, paths):
        """Import slides from chosen files, dropped files, or whole folders.

        Only archives create entries: a log sidecar on its own has nothing to
        attach to, so it is reported as skipped rather than listed. Archives
        already in the project are skipped too, and an entry that was imported
        without its log picks one up if it has since appeared.
        """
        npz_paths, log_paths, rejected = collect_slide_paths(paths)
        known = {e.npz_path: e for e in self.entries}
        added = []

        for npz_path in npz_paths:
            npz_path = os.path.abspath(npz_path)
            name = os.path.splitext(os.path.basename(npz_path))[0]
            log_path = log_paths.get(name) or find_log_file(npz_path)

            if npz_path in known:
                # already imported: fill in a log that has turned up since
                if log_path and not known[npz_path].log_path:
                    known[npz_path].log_path = os.path.abspath(log_path)
                else:
                    rejected.append((npz_path, "already in project"))
                continue

            entry = SlideEntry(npz_path, log_path)
            self.entries.append(entry)
            known[npz_path] = entry
            added.append(entry)

        # a log whose archive is nowhere to be found cannot be listed
        matched = {e.name for e in self.entries}
        rejected += [(p, "no matching .npz") for n, p in log_paths.items()
                     if n not in matched]

        self.entries.sort(key=lambda e: e.name)
        self.save()
        return added, rejected

    # -- repairing moved slides --------------------------------------------

    def missing_entries(self):
        return [e for e in self.entries if not os.path.exists(e.npz_path)]

    def relocate_entry(self, entry, new_npz):
        """Point one entry at a found archive, and take its log with it."""
        entry.npz_path = os.path.abspath(new_npz)
        entry.log_path = find_log_file(entry.npz_path)
        self.save()

    def relocate_prefix(self, old_prefix, new_prefix):
        """Repoint every other missing entry that moved the same way.

        One located slide usually fixes a whole cohort, since a moved or
        renamed drive changes the same leading path for all of them.
        """
        fixed = []
        for entry in self.missing_entries():
            if not entry.npz_path.startswith(old_prefix):
                continue
            candidate = new_prefix + entry.npz_path[len(old_prefix):]
            if os.path.exists(candidate):
                entry.npz_path = candidate
                entry.log_path = find_log_file(candidate)
                fixed.append(entry)

        if fixed:
            self.save()
        return fixed

    def remove_entries(self, entries):
        """Drop slides from the project, with the files the project owns.

        Only the project's own files go: the annotation GeoJSON and the QC row.
        The .npz and its log sit outside the project and are left alone.
        """
        for entry in list(entries):
            path = self.annotation_path(entry)
            if os.path.exists(path):
                os.remove(path)
            self.qc.pop(entry.name, None)
            if entry in self.entries:
                self.entries.remove(entry)

        self.save()
        self.write_qc()

    def set_user(self, user):
        self.user = user
        self.save()
