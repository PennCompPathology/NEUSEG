#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""SANA ROI Annotator.

A Qt viewer for drawing cortical ROIs on preprocessed whole-slide images.

The NEUSEG pipeline writes one compressed archive (`<slide>.npz`) per slide
holding the thumbnail, the soma feature heatmap and the tissue/GM/WM masks.
This tool walks a directory of those archives, renders each slide, and lets the
user lay down GM and deep-WM ROIs, which are written back out as GeoJSON beside
the archive.

Layout of this file:

    1. Annotation      - the ROI data model and its geometry
    2. Canvas          - the drawing surface and click-to-ROI logic
    3. SlideAnnotator  - the main window: slide I/O, overlays, zoom, actions
    4. Qt widgets      - dock panels and their controls
    5. NEUSEG viewer   - re-running the GM/WM segmentation and its GMM plot
    6. GeoJSON I/O     - reading and writing annotation files
    7. main            - command line entry point

Slides are always reached through a project (see project.py), which records
where each .npz lives and holds the annotations made against it.

Usage:
    python pdnl_annotator.py [project_dir] [start_idx]
"""

import os
import sys
import json
import fnmatch
import datetime
import re
import functools
import time

import geojson
import numpy as np
import cv2

# matplotlib picks its own Qt binding, and tries PyQt6 before PySide6. On a
# machine with both installed the histogram canvas would come back as a PyQt6
# widget, which no PySide6 layout will accept. Pin it before the backend loads.
os.environ["QT_API"] = "pyside6"

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap

from PySide6.QtCore import (Qt, QPoint, QPointF, QLine, QTimer, Signal,
                            QThread, QEvent, QRect)
from PySide6.QtGui import (QImage, QPixmap, QPalette, QPainter, QAction, QPen,
                           QFont, QColor, QBrush, QCursor, QIcon)
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QButtonGroup, QCheckBox, QComboBox,
    QDockWidget, QFileDialog, QHBoxLayout, QInputDialog, QLabel, QLineEdit,
    QDialog, QListWidget, QListWidgetItem, QMainWindow, QMenu, QMessageBox,
    QProgressBar, QPushButton, QRadioButton, QScrollArea, QSizePolicy,
    QSlider, QSpinBox, QStyle, QToolBar, QVBoxLayout, QWidget,
)

import pdnl_sana as sana
import pdnl_sana.geo
import pdnl_sana.image
import pdnl_sana.interpolate
import pdnl_sana.logging

# this runs as a script, so make sure its own folder is importable no matter
# which directory it was launched from
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import project as prj

# Cursor snapping walks the tissue/GM/WM contours so clicks land exactly on a
# segmentation vertex. It is switched OFF: the one-click `Canvas.get_roi` path
# derives the ROI corners from the masks directly, so manual snapping is not
# needed. Set to True to re-enable it along with the two dock checkboxes it
# drives (`snap_to_seg`, `annotate_adjacent_wm`), which are disabled to match.
SNAP_TO_SEGMENTATION = False

# values in the combined segmentation mask handed to Canvas:
# 0 background, 1 grey matter, 2 white matter
BACKGROUND, WHITE_MATTER = 0, 2

# microns-per-pixel scaling used to turn the GM width spinbox (microns) into a
# pixel half-width at thumbnail resolution
THUMBNAIL_DOWNSAMPLE = 16
MICRONS_PER_PIXEL = 0.5045

ZOOM_STEP = 1.25

# the slide name shares the toolbar with two legends, so it cannot be huge
SLIDE_NAME_POINT_SIZE = 15

# the two channels stored in feature_heatmap, in channel order
FEATURE_CHANNELS = (("Nuclei Density", 0), ("Nuclei Size", 1))

# the heatmap ramp as a 256-entry RGB lookup table, so it can be any
# matplotlib colormap with no BGR conversion
DEFAULT_HEATMAP_OPACITY = 50
HEATMAP_LUT = (matplotlib.colormaps["Reds"](np.linspace(0, 1, 256))[:, :3]
               * 255).astype(np.uint8)

# the same ramp as a 256x1 image, for the on-canvas colour bar. The buffer has
# to outlive the QImage, so it is kept here rather than built on each paint.
HEATMAP_LUT_BYTES = HEATMAP_LUT.tobytes()

# how often the progress bar is nudged between the worker's stage reports,
# and the run length assumed before one has been timed on this machine
PROGRESS_TICK_MS = 200
DEFAULT_RUN_SECONDS = 21.0

# the key drawn over the top-right of the slide while the heatmap is on
LEGEND_SIZE = (190, 48)
LEGEND_PAD = 10

# the histogram samples every Nth tissue pixel so it stays cheap to redraw
HISTOGRAM_STRIDE = 4
HISTOGRAM_BINS = 90

# bars inside the kept window match the range slider; the rest are greyed
HISTOGRAM_IN_COLOUR = "#0d47a1"
HISTOGRAM_OUT_COLOUR = "#c4c4c4"
HISTOGRAM_EDGE = "#4a4a4a"

# The window opens at this share of the screen, in this shape, and then stays
# put: loading a slide only rescales the image inside it. Sizing by ratio
# rather than in pixels keeps the same proportions on any monitor - a fixed
# size ends up filling a laptop screen entirely. The user is still free to drag
# the window or its edges afterwards, and nothing moves it back.
WINDOW_SCREEN_FRACTION = 0.72
WINDOW_ASPECT = 16 / 10

# how each project entry is marked in the slide list: glyph, colour, tooltip.
# an entry only exists once an .npz has been imported, so a bare log sidecar
# never reaches this table
# a complete import is the expected case, so it is left unmarked; the others
# get a glyph, drawn in the ordinary text colour like everything else
ENTRY_STATUS_DISPLAY = {
    prj.STATUS_OK:      (" ", "NPZ and _log.pkl both imported"),
    prj.STATUS_NO_LOG:  ("⚠", "NPZ only - no _log.pkl found"),
    prj.STATUS_MISSING: ("✗", "NPZ missing from its recorded path"),
}

# Boundary colours, as RGB, matching the GMM plot so the two views agree.
CONTOUR_COLOURS = {"gm_wm": (27, 127, 59), "gm_csf": (106, 27, 154),
                   "wm_csf": (0, 0, 0)}
CONTOUR_WARNING_COLOUR = (211, 47, 47)
CONTOUR_WIDTH = 5

# the region fills reuse the boundary colours, so an outline and the area it
# encloses read as the same class
MASK_COLOURS = {"gm": CONTOUR_COLOURS["gm_csf"], "wm": CONTOUR_COLOURS["gm_wm"]}
DEFAULT_MASK_OPACITY = 35

# the kept span of the range slider
RANGE_SLIDER_COLOUR = "#0d47a1"

CONTOUR_LEGEND = "".join(
    '<span style="color:#%02x%02x%02x">&#9632;</span> %s &nbsp;'
    % (*CONTOUR_COLOURS[key], label)
    for key, label in (("gm_csf", "GM/CSF"), ("gm_wm", "GM/WM"),
                       ("wm_csf", "WM/CSF"))) + "&nbsp;&nbsp;"

# QC shows as a small colour chip in its own column, left of the row text, so
# the glyphs and the slide name keep the theme's ordinary colour
QC_DISPLAY = {"Minor errors": "#ffd600", "Fail": "#d32f2f"}

# bad tissue strikes the row through instead of taking the chip: the chip
# carries the grade, so the two signals stay independent. Qt draws a font
# strikeout in the text colour, which is why the name goes red with it.
QC_BAD_TISSUE_COLOUR = "#d32f2f"
QC_ICON_SIZE = 11

# second chip slot: this slide has ratings recorded elsewhere
RATED_COLOUR = "#1e88e5"

# the one-line tally above the slide list: glyph, colour, and what it counts.
# problems first, then progress
SUMMARY_PARTS = (("no_log",  "\u26a0", None,         "no _log.pkl"),
                 ("missing", "\u2717", None,         "archive not found"),
                 ("minor",   "\u25a0", "#c8a200",    "QC: minor errors"),
                 ("fail",    "\u25a0", "#d32f2f",    "QC: fail"),
                 ("bad",     "\u2298", "#d32f2f",    "bad tissue"),
                 ("rated",   "\u25a0", RATED_COLOUR, "has ratings"))

# ROI names the app generated itself. Only these are renumbered on delete -
# anything hand-typed is left as the user wrote it.
AUTO_ROI_NAME = re.compile(r"^(GM|WM)_\d+$")

# follows the status glyph on slides that already have ROIs saved
ANNOTATED_GLYPH = "✎"

# half-length of the crosshair arms marking a hovered GMM scatter point, in
# screen pixels so the marker stays the same size at any zoom
HIGHLIGHT_ARM = 9
HIGHLIGHT_WIDTH = 2

# the GMM is fit on every tissue pixel - far too many to plot, so the scatter
# shows a random sample of them, which preserves the shape of the cloud
SCATTER_SAMPLE = 25000

# the posterior runs 0 = white matter to 1 = grey matter, through a pale middle
# so the pixels the model is unsure about stand out as washed rather than mixed
GMM_COLORMAP = LinearSegmentedColormap.from_list(
    "wm_gm", ["#1b7f3b", "#eeeeee", "#6a1b9a"])

# shown at the right-hand end of the navigation toolbar; keep in step with
# keyPressEvent
SHORTCUT_LEGEND = ("A / W  annotate \u00b7 ESC  cancel \u00b7 H  heatmap \u00b7 "
                   "D / S  density / size \u00b7 M  mask \u00b7 C  contours \u00b7 "
                   "< >  slide  ")


@functools.lru_cache(maxsize=16)
def row_icon(qc_colour, rated):
    """Up to two squares in fixed slots, so they line up down the column.

    Slot one is the QC verdict, slot two the ratings marker; either may be
    empty. Cached, since every row asks for one of a handful of combinations.
    """
    step = QC_ICON_SIZE + 3
    pixmap = QPixmap(2 * step, QC_ICON_SIZE)
    pixmap.fill(Qt.transparent)

    qp = QPainter(pixmap)
    qp.setPen(QPen(QColor("#6b6b6b"), 1))
    for slot, colour in ((0, qc_colour), (1, RATED_COLOUR if rated else None)):
        if colour:
            qp.setBrush(QColor(colour))
            qp.drawRect(slot * step + 1, 1, QC_ICON_SIZE - 3, QC_ICON_SIZE - 3)
    qp.end()
    return QIcon(pixmap)


def entry_row(entry, n_annotations, qc=prj.QC_EMPTY, rated=False):
    """Text, QC chip and tooltip for one row of the slide list.

    The three signals stay on separate channels so none hides another: import
    state and annotation count are glyphs in the text, the QC verdict is a
    colour chip in the icon column, and the text itself is never recoloured.
    """
    glyph, tip = ENTRY_STATUS_DISPLAY[entry.status]
    mark = f"{ANNOTATED_GLYPH}{n_annotations}" if n_annotations else ""
    done = f"{n_annotations} ROI(s) saved" if n_annotations else "no ROIs yet"

    colour = QC_DISPLAY.get(qc["status"])
    flags = " + ".join(filter(None, [qc["status"],
                                     "bad tissue" if qc["bad_tissue"] else ""]))
    if flags:
        tip = f"QC: {flags}\n{tip}"
    if rated:
        tip = f"Has ratings\n{tip}"

    return (f"{glyph} {mark:<3s} {entry.name}",
            row_icon(colour, rated) if (colour or rated) else QIcon(),
            f"{tip}\n{done}\n{entry.npz_path}",
            qc["bad_tissue"])


# ---------------------------------------------------------------------------
# 1. Annotation
# ---------------------------------------------------------------------------

class Annotation:
    """One ROI: a GM band, an adjacent/deep WM box, or both.

    A GM ROI is bounded by four curves - the CSF/GM boundary, the GM/WM
    boundary, and the two side walls joining their endpoints. A WM ROI is a
    simple four-point polygon.
    """

    def __init__(self, is_gm, is_wm):
        # CSF/GM boundary endpoints, and the contour they were sampled from
        self.csf0 = None
        self.csf1 = None
        self.csf_poly = None
        self.csf0_idx = None
        self.csf1_idx = None

        # GM/WM boundary endpoints, and the contour they were sampled from
        self.wm0 = None
        self.wm1 = None
        self.wm_poly = None
        self.wm0_idx = None
        self.wm1_idx = None

        # the two extra corners that close off a WM box
        self.wm2 = None
        self.wm3 = None

        self.is_gm = is_gm
        self.is_wm = is_wm
        self.saved = False
        self.done = False
        self.name = ""

    def save(self):
        """Freeze the clicked points into the final ROI curves/polygons."""
        self.saved = True

        if self.is_gm:
            self._build_gm_roi()
            if self.is_wm:
                self._build_adjacent_wm_roi()
        else:
            # deep WM: a plain quadrilateral through the four clicked corners
            self.wm_roi = sana.geo.Polygon(
                [self.wm0.x(), self.wm1.x(), self.wm2.x(), self.wm3.x()],
                [self.wm0.y(), self.wm1.y(), self.wm2.y(), self.wm3.y()],
            ).astype(float).connect()

    def _boundary_curve(self, poly, i0, i1, p0, p1):
        """Smooth the contour arc between two vertices.

        Falls back to the straight chord when there is no contour to follow, or
        when the polynomial fit fails (near-vertical or degenerate arcs).
        """
        if poly is None:
            return sana.geo.Curve([p0.x(), p1.x()], [p0.y(), p1.y()]).astype(float)

        arc = poly.slice_shortest(i0, i1).astype(float)
        smoothed = sana.interpolate.fit_rotated_polynomial(arc, 3, 20)
        if smoothed is None:
            return poly[np.array([i0, i1])].to_curve()
        return smoothed

    def _build_gm_roi(self):
        """Orient the four GM boundary curves into a consistent winding.

        Downstream sampling assumes CSF runs left-to-right along the top, GM/WM
        right-to-left along the bottom, and the walls run between them. The
        curves are rotated flat, flipped into that order, then rotated back.
        """
        csf = self._boundary_curve(self.csf_poly, self.csf0_idx, self.csf1_idx,
                                   self.csf0, self.csf1)
        gm = self._boundary_curve(self.wm_poly, self.wm0_idx, self.wm1_idx,
                                  self.wm0, self.wm1)

        # side walls connecting the CSF endpoints to the GM/WM endpoints
        s0 = sana.geo.curve_like(csf, [self.csf0.x(), self.wm0.x()],
                                 [self.csf0.y(), self.wm0.y()]).astype(float)
        s1 = sana.geo.curve_like(csf, [self.csf1.x(), self.wm1.x()],
                                 [self.csf1.y(), self.wm1.y()]).astype(float)

        ctr = sana.geo.point_like(csf, 0, 0)
        curves = [csf, gm, s0, s1]

        # rotate so the CSF boundary lies horizontal
        angle = csf.get_angle()
        [x.rotate(ctr, -angle) for x in curves]

        # make sure CSF is above GM/WM, not below
        if np.mean(csf[:, 1]) > np.mean(gm[:, 1]):
            angle += 180
            [x.rotate(ctr, 180) for x in curves]

        # whichever wall sits further left is the left wall
        if np.mean(s0[:, 0]) < np.mean(s1[:, 0]):
            left, right = s0, s1
        else:
            left, right = s1, s0

        # walk the ROI boundary consistently: CSF left-to-right, down the right
        # wall, GM/WM right-to-left, up the left wall
        if csf[0, 0] > csf[-1, 0]:
            csf = csf[::-1]
        if right[0, 1] > right[-1, 1]:
            right = right[::-1]
        if gm[0, 0] < gm[-1, 0]:
            gm = gm[::-1]
        if left[0, 1] < left[-1, 1]:
            left = left[::-1]

        [x.rotate(ctr, angle) for x in curves]

        self.csf_gm_seg = csf
        self.gm_wm_seg = gm
        self.left_wall = left
        self.right_wall = right

    def _build_adjacent_wm_roi(self):
        """Close the GM/WM boundary into a WM polygon with the two extra corners.

        The corners can be given in either order, so both windings are built and
        the one enclosing more area wins (the other self-intersects).
        """
        candidates = [
            sana.geo.Polygon(
                [*self.gm_wm_seg[:, 0], a.x(), b.x()],
                [*self.gm_wm_seg[:, 1], a.y(), b.y()],
            ).astype(float)
            for a, b in ((self.wm2, self.wm3), (self.wm3, self.wm2))
        ]
        first, second = candidates
        winner = first if first.get_area() > second.get_area() else second
        self.wm_roi = winner.connect()


# ---------------------------------------------------------------------------
# 2. Canvas
# ---------------------------------------------------------------------------

class Canvas(QLabel):
    """The slide image plus the annotation overlay drawn on top of it."""

    def __init__(self, mask):
        super().__init__()
        self.point = None           # cursor position in image coordinates
        self.idx = None             # index of the snapped contour vertex
        self.poly = None            # contour the cursor is snapped to
        self.annotations = []
        self.current_annotation = None
        self.radius = 4
        self.scale_factor = 1.0
        self.mask = mask            # combined tissue/WM mask, values 0/1/2
        self.gm_polys = []
        self.highlight = None       # set from the GMM window's hover
        self.is_annotating = False  # picks the cursor shape
        self.heatmap_legend = None  # (title, low, high) while the heatmap is on
        self.selected = set()       # indices highlighted from the ROI list

    def reset_annotations(self):
        self.annotations = []
        self.current_annotation = None
        self.update()

    # -- ROI construction ---------------------------------------------------

    @staticmethod
    def _cast_ray(mask, ctr, theta, direction, csf, wm):
        """Step outward from `ctr` along `theta` until a boundary is crossed.

        Returns the (csf, wm) pair updated with whichever boundary was hit
        first; already-found boundaries are left alone so the opposite ray can
        fill in the other one.
        """
        h, w = mask.img.shape[:2]
        r = 0
        while True:
            r += direction
            x = int(r * np.cos(theta) + ctr[0])
            y = int(r * np.sin(theta) + ctr[1])
            if not (0 <= x < w and 0 <= y < h):
                return csf, wm
            if csf is None and mask.img[y, x] == BACKGROUND:
                return (x, y), wm
            if wm is None and mask.img[y, x] == WHITE_MATTER:
                return csf, (x, y)

    def get_shortest(self, gm, ctr, n_angles=360):
        """Find the shortest CSF-to-WM chord through `ctr`.

        Sweeps rays through a half turn, casting both ways along each so a
        chord is found even when `ctr` sits off-centre in the ribbon, and keeps
        the shortest - the local cortical thickness.
        """
        dists, pts = [], []
        for theta in np.linspace(0, np.pi, n_angles):
            csf, wm = self._cast_ray(gm, ctr, theta, +1, None, None)
            csf, wm = self._cast_ray(gm, ctr, theta, -1, csf, wm)
            if csf is None or wm is None:
                continue
            dists.append(np.sqrt((csf[0] - wm[0]) ** 2 + (csf[1] - wm[1]) ** 2))
            pts.append([csf, wm])

        return pts[np.argmin(dists)]

    @staticmethod
    def _nearest_contour(polys, p0, p1):
        """Pick the contour closest to both points, and the vertices nearest each.

        Scoring on the summed distance keeps the pair on a single contour, so an
        ROI never straddles two disconnected pieces of cortex.
        """
        dists, idxs = [], []
        for poly in polys:
            d0 = np.sqrt((poly[:, 0] - p0[0]) ** 2 + (poly[:, 1] - p0[1]) ** 2)
            d1 = np.sqrt((poly[:, 0] - p1[0]) ** 2 + (poly[:, 1] - p1[1]) ** 2)
            idxs.append([np.argmin(d0), np.argmin(d1)])
            dists.append(np.min(d0) + np.min(d1))

        best = np.argmin(dists)
        return polys[best], idxs[best][0], idxs[best][1]

    def get_roi(self, gm, ctr, l):
        """Build a GM ROI from a single click at `ctr`, `l` pixels to each side.

        Walks out laterally from the click, measures the cortical ribbon at both
        shoulders, then snaps those four points onto the GM contours to get the
        vertex range each boundary should follow.
        """
        # the spine of the ROI: the cortical thickness at the click itself
        csf, wm = self.get_shortest(gm, ctr)

        # step perpendicular to the spine to find the ROI's lateral extent
        th = np.arctan2((wm[1] - csf[1]), (wm[0] - csf[0]))
        thp = th + np.pi / 2
        p0 = (-l * np.cos(thp) + ctr[0], -l * np.sin(thp) + ctr[1])
        p1 = (l * np.cos(thp) + ctr[0], l * np.sin(thp) + ctr[1])

        # the ribbon at each shoulder gives the ROI's four corners
        csf0, wm0 = self.get_shortest(gm, p0)
        csf1, wm1 = self.get_shortest(gm, p1)

        # snap the corners onto the traced contours so the boundaries can follow
        # the real tissue edge rather than a straight line
        csf_poly, csf_idx_0, csf_idx_1 = self._nearest_contour(self.gm_polys, csf0, csf1)
        wm_poly, wm_idx_0, wm_idx_1 = self._nearest_contour(self.gm_polys, wm0, wm1)

        return csf_poly, csf_idx_0, csf_idx_1, wm_poly, wm_idx_0, wm_idx_1

    def select_point(self):
        """Commit the cursor position as the next control point of the ROI."""
        a = self.current_annotation
        if a is None or self.point is None:
            return

        if a.is_gm:
            # one click is enough: the whole GM ROI is derived from the masks
            half_width = self.gm_width // (2 * THUMBNAIL_DOWNSAMPLE * MICRONS_PER_PIXEL)
            csf_poly, csf0, csf1, wm_poly, wm0, wm1 = self.get_roi(
                self.mask, (self.point.x(), self.point.y()), l=half_width)

            a.csf_poly, a.csf0_idx, a.csf1_idx = csf_poly, csf0, csf1
            a.csf0 = QPoint(*csf_poly[csf0])
            a.csf1 = QPoint(*csf_poly[csf1])

            a.wm_poly, a.wm0_idx, a.wm1_idx = wm_poly, wm0, wm1
            a.wm0 = QPoint(*wm_poly[wm0])
            a.wm1 = QPoint(*wm_poly[wm1])

            a.done = True

        elif a.is_wm:
            # deep WM is drawn by hand, one corner per click
            for attr in ("wm0", "wm1", "wm2", "wm3"):
                if getattr(a, attr) is None:
                    setattr(a, attr, self.point)
                    a.done = (attr == "wm3")
                    break

        self.update()

    # -- drawing primitives -------------------------------------------------

    def draw_circle(self, qp, p0):
        if p0 is None:
            return
        qp.drawEllipse(p0 * self.scale_factor, self.radius, self.radius)

    def draw_line(self, qp, p0, p1):
        if p0 is None or p1 is None:
            return
        qp.drawLine(QLine(p0 * self.scale_factor, p1 * self.scale_factor))

    def draw_curve(self, qp, c):
        for i in range(len(c) - 1):
            self.draw_line(qp, QPoint(*c[i]), QPoint(*c[i + 1]))

    # TODO: arc should be stored somewhere and we just plot it
    def draw_arc(self, qp, poly, i0, i1):
        """Draw the contour arc between two vertices, smoothed where possible."""
        if poly is None or i0 is None or i1 is None or i0 == i1:
            return
        arc = poly.slice_shortest(i0, i1)
        smoothed = sana.interpolate.fit_rotated_polynomial(arc, 3, 10)
        self.draw_curve(qp, arc if smoothed is None else smoothed)

    def draw_crosshair(self, qp, p):
        """A black cross on a white halo, at a fixed screen size.

        Used both for the cursor while inspecting and for the point the GMM
        window is pointing at, so the two read as the same thing.
        """
        if p is None:
            return
        # draw_line scales by the zoom, so undo it to keep a constant size
        arm = int(HIGHLIGHT_ARM / max(self.scale_factor, 1e-6))
        arms = ((QPoint(p.x() - arm, p.y()), QPoint(p.x() + arm, p.y())),
                (QPoint(p.x(), p.y() - arm), QPoint(p.x(), p.y() + arm)))

        # the halo keeps it legible on dark tissue and on the feature heatmap
        for colour, width in ((Qt.white, HIGHLIGHT_WIDTH + 2), (Qt.black, HIGHLIGHT_WIDTH)):
            qp.setPen(QPen(colour, width))
            for a, b in arms:
                self.draw_line(qp, a, b)
        qp.setPen(QPen(Qt.green, self.radius))

    def draw_text(self, qp, x, y, text):
        w = len(text) * self.radius
        h = self.radius
        qp.drawText(QPoint((x - w // 2) * self.scale_factor,
                           (y - h // 2) * self.scale_factor), text)

    # -- annotation rendering -----------------------------------------------

    # TODO: this should be drawn on the image itself once saved
    # TODO: then reset the image if reseting annotations
    def draw_annotation(self, qp, a, selected=False):
        """Saved ROIs render as a closed red outline; in-progress ones as green
        control points and the edges between them. A selected ROI is drawn in
        cyan so the list and the slide agree on which one is which."""
        if a.saved:
            qp.setPen(QPen(Qt.cyan if selected else Qt.red, self.radius))
            self._draw_saved_annotation(qp, a)
            qp.setPen(QPen(Qt.green, self.radius))
        else:
            self._draw_pending_annotation(qp, a)

    def _draw_saved_annotation(self, qp, a):
        if a.is_gm:
            roi = sana.geo.polygon_like(a.csf_gm_seg, *np.concatenate(
                [a.csf_gm_seg, a.right_wall, a.gm_wm_seg, a.left_wall], axis=0).T)
            self.draw_curve(qp, roi)
            self.draw_text(qp, *roi.get_centroid(), a.name)
        if a.is_wm:
            self.draw_curve(qp, a.wm_roi)
            self.draw_text(qp, *a.wm_roi.get_centroid(), a.name)

    def _draw_pending_annotation(self, qp, a):
        for p in (a.csf0, a.csf1, a.wm0, a.wm1, a.wm2, a.wm3):
            self.draw_circle(qp, p)

        # the two side walls
        self.draw_line(qp, a.csf0, a.wm0)
        self.draw_line(qp, a.csf1, a.wm1)

        # CSF and GM/WM boundaries follow a contour when one was snapped to
        if a.csf_poly is not None:
            self.draw_arc(qp, a.csf_poly, a.csf0_idx, a.csf1_idx)
        else:
            self.draw_line(qp, a.csf0, a.csf1)

        if a.is_gm and a.wm_poly is not None:
            self.draw_arc(qp, a.wm_poly, a.wm0_idx, a.wm1_idx)
        else:
            self.draw_line(qp, a.wm0, a.wm1)

        # the WM box corners
        self.draw_line(qp, a.wm1, a.wm2)
        self.draw_line(qp, a.wm2, a.wm3)
        self.draw_line(qp, a.wm3, a.wm0)

    def _draw_rubber_band(self, qp, a):
        """Draw the edge that follows the cursor for the next unplaced point."""
        if a.is_gm:
            # first wall
            if a.csf0 is not None and a.wm0 is None:
                self.draw_line(qp, a.csf0, self.point)

            # CSF boundary
            if a.csf0 is not None and a.wm0 is not None and a.csf1 is None:
                if a.csf_poly is not None:
                    self.draw_arc(qp, a.csf_poly, a.csf0_idx, self.idx)
                else:
                    self.draw_line(qp, a.csf0, self.point)

            # second wall
            if a.csf1 is not None and a.wm1 is None:
                self.draw_line(qp, a.csf1, self.point)

            # GM/WM boundary
            if a.wm0 is not None and a.csf1 is not None and a.wm1 is None:
                if a.wm_poly is not None:
                    self.draw_arc(qp, a.wm_poly, a.wm0_idx, self.idx)
                else:
                    self.draw_line(qp, a.wm0, self.point)

            # the adjacent WM box hanging off the GM/WM boundary
            if a.is_wm and a.wm1 is not None and a.wm2 is None:
                self.draw_line(qp, a.wm1, self.point)
            if a.is_wm and a.wm2 is not None and a.wm3 is None:
                self.draw_line(qp, a.wm2, self.point)
                self.draw_line(qp, a.wm0, self.point)

        elif a.is_wm:
            # deep WM box, corner by corner
            if a.wm0 is not None and a.wm1 is None:
                self.draw_line(qp, a.wm0, self.point)
            if a.wm1 is not None and a.wm2 is None:
                self.draw_line(qp, a.wm1, self.point)
            if a.wm2 is not None and a.wm3 is None:
                self.draw_line(qp, a.wm2, self.point)
                self.draw_line(qp, a.wm0, self.point)

    def draw_heatmap_legend(self, qp):
        """Name the channel on screen, with its colour ramp, top right.

        Anchored to the visible part of the canvas rather than the image, so
        it stays in view when zoomed in and scrolled.
        """
        if self.heatmap_legend is None:
            return
        title, low, high = self.heatmap_legend

        area = self.visibleRegion().boundingRect()
        if area.isEmpty():
            area = self.rect()
        w, h = LEGEND_SIZE
        x = area.right() - w - LEGEND_PAD
        y = area.top() + LEGEND_PAD

        qp.save()
        qp.setPen(Qt.NoPen)
        qp.setBrush(QColor(255, 255, 255, 220))
        qp.drawRoundedRect(x, y, w, h, 5, 5)

        font = qp.font()
        font.setPointSize(9)
        font.setBold(True)
        qp.setFont(font)
        qp.setPen(QColor("#222222"))
        qp.drawText(QRect(x, y + 3, w, 15), Qt.AlignmentFlag.AlignHCenter, title)

        ramp = QRect(x + 10, y + 21, w - 20, 9)
        qp.drawImage(ramp, QImage(HEATMAP_LUT_BYTES, 256, 1, 3 * 256,
                                  QImage.Format.Format_RGB888))
        qp.setPen(QColor("#777777"))
        qp.setBrush(Qt.NoBrush)
        qp.drawRect(ramp)

        font.setBold(False)
        font.setPointSize(7)
        qp.setFont(font)
        qp.setPen(QColor("#333333"))
        labels = QRect(x + 10, y + 32, w - 20, 12)
        qp.drawText(labels, Qt.AlignmentFlag.AlignLeft, low)
        qp.drawText(labels, Qt.AlignmentFlag.AlignRight, high)
        qp.restore()

    def paintEvent(self, event):
        super().paintEvent(event)

        qp = QPainter(self)
        qp.setPen(QPen(Qt.green, self.radius))
        qp.setBrush(Qt.black)

        for i, annotation in enumerate(self.annotations):
            self.draw_annotation(qp, annotation, selected=i in self.selected)

        # the cursor shape is the mode indicator: the green ring means a click
        # will place a control point, the crosshair means you are only looking
        if self.point is not None:
            if self.is_annotating:
                self.draw_circle(qp, self.point)
            else:
                self.draw_crosshair(qp, self.point)

        # where the GMM window is pointing, when the mouse is over there instead
        self.draw_crosshair(qp, self.highlight)

        a = self.current_annotation
        if a is not None:
            self.draw_annotation(qp, a)
            self._draw_rubber_band(qp, a)

        # last, so it sits above the slide and the annotations
        self.draw_heatmap_legend(qp)


# ---------------------------------------------------------------------------
# 3. SlideAnnotator
# ---------------------------------------------------------------------------

class SlideAnnotator(QMainWindow):

    # the pipeline writes one compressed archive per slide instead of a
    # directory of loose .npy/.png files; these are the arrays read from it
    ARRAY_KEYS = {
        "Thumbnail": "thumbnail",
        "Features": "feature_heatmap",
        "GM Mask": "gm_mask",
        "WM Mask": "wm_mask",
        "Tissue Mask": "tissue_mask",
    }

    def __init__(self, project=None, start_idx=0, do_ask_name=True):
        super().__init__()

        style = QApplication.style()
        self.TITLEBAR_HEIGHT = style.pixelMetric(QStyle.PixelMetric.PM_TitleBarHeight)

        self.scale_factor = 1.0
        self.source_frame = None

        # set once a project is adopted; no slide is loaded until then
        self.project = None
        self.slide_entry = None
        self.slide_name = None

        # drives the progress bar between the worker's stage reports
        self.progress_timer = QTimer(self)
        self.progress_timer.timeout.connect(self.tick_neuseg_progress)
        self.expected_run_seconds = DEFAULT_RUN_SECONDS
        self.run_start = self.stage_start = 0.0
        self.stage_from, self.stage_to = 0, 100

        # populated per slide by cache_feature_stats
        self.features = None
        self.stored_gm_mask = None
        self.boundaries = {}
        self.gm_bool = self.wm_bool = None
        self.gmm_result = None
        self.gmm_window = None
        self.qc_window = None

        # slider position per (slide, channel), for this session only: the
        # window suits one slide's tissue, so it should not follow you to the
        # next one. Nothing is written to the project.
        self.feature_windows = {}
        self.window_key = None
        self.last_gmm_pixel = None
        self.feature_values = {}
        self.feature_limits = {}
        self.segmentation_mask = None

        self.canvas = Canvas(self.segmentation_mask)
        self.canvas.setBackgroundRole(QPalette.ColorRole.Base)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.canvas.setScaledContents(True)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidget(self.canvas)
        self.scroll_area.setVisible(False)
        self.scroll_area.setAlignment(
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignHCenter)
        self.setCentralWidget(self.scroll_area)

        self.setWindowTitle("SANA ROI Annotator")

        self.init_docks()
        self.init_navigation()
        self.create_actions()
        self.create_menus()

        # after the docks: they set the minimum size the window can honour
        self.init_window_geometry()

        self.is_annotating = False
        self.do_ask_name = do_ask_name

        # hotkeys are routed through a filter, not just keyPressEvent
        QApplication.instance().installEventFilter(self)

        # show before loading so the scroll area has a real size to fit the
        # first slide into
        self.show()
        self.setFocus()

        # everything is done inside a project, so ask for one up front; the
        # window still opens if the user declines, ready for File > New Project
        if project is None:
            project = self.choose_project()
        if project is None:
            self.clear_slide_state()
        else:
            self.set_project(project, start_idx)

    @staticmethod
    def window_geometry(available, minimum):
        """Centred rect at a fixed share and shape, never below `minimum`.

        Split out from the call below so the arithmetic can be checked against
        any screen size without a real window.
        """
        w = available.width() * WINDOW_SCREEN_FRACTION
        h = w / WINDOW_ASPECT

        # take the fraction of whichever dimension binds first, so the shape
        # holds and the window never exceeds the screen
        if h > available.height() * WINDOW_SCREEN_FRACTION:
            h = available.height() * WINDOW_SCREEN_FRACTION
            w = h * WINDOW_ASPECT

        # the docks impose a floor; clamp here rather than letting Qt silently
        # enlarge the window and throw the centring off
        w = max(int(w), minimum.width())
        h = max(int(h), minimum.height())

        return (max(available.x(), available.x() + (available.width() - w) // 2),
                max(available.y(), available.y() + (available.height() - h) // 2),
                w, h)

    def init_window_geometry(self):
        """Open centred on the screen being used, at a fixed share and shape.

        The screen under the cursor, not the primary one, so launching on a
        laptop beside a larger display is sized for the laptop.
        `availableGeometry` is in global coordinates and its origin is not
        (0, 0), so the centring is relative to that origin.
        """
        screen = QApplication.screenAt(QCursor.pos()) or QApplication.primaryScreen()
        self.setGeometry(*self.window_geometry(screen.availableGeometry(),
                                               self.minimumSizeHint()))

    def message_dialog(self, title, text):
        dlg = QMessageBox(self)
        dlg.setWindowTitle(title)
        dlg.setText(text)
        dlg.exec()

    def error_dialog(self, text):
        """Show a modal error box. Used for every user-facing failure."""
        self.message_dialog("Error", text)

    def info_dialog(self, text):
        self.message_dialog("Import", text)

    # -- input modes --------------------------------------------------------

    def toggle_mouse_tracking(self, state):
        self.setMouseTracking(state)
        self.canvas.setMouseTracking(state)
        self.scroll_area.setMouseTracking(state)
        self.canvas.setFocus()
        self.canvas.point = None
        self.canvas.update()

    def toggle_is_annotating(self, state=None):
        """Set annotating mode, or flip it when no explicit state is given."""
        self.is_annotating = (not self.is_annotating) if state is None else state
        self.canvas.is_annotating = self.is_annotating
        self.update_mouse_tracking()

    def update_mouse_tracking(self):
        """Qt only delivers mouse moves while tracking is on or a button is
        held, so it has to be on both for annotating and for the GMM plot's
        live marker to follow the cursor over the slide."""
        self.toggle_mouse_tracking(
            self.is_annotating
            or (self.gmm_window is not None and self.gmm_window.isVisible()))

    def eventFilter(self, obj, event):
        """Let the hotkeys fire from anywhere in the main window.

        Lists, spin boxes and combo boxes consume plain letters for their own
        type-ahead, so a key pressed after clicking the project panel never
        reached keyPressEvent. Real text entry still wins - otherwise a comment
        could not contain an "a" - and other windows are left alone.
        """
        if event.type() == QEvent.Type.KeyPress:
            focus = QApplication.focusWidget()
            if (focus is not None and focus.window() is self
                    and not isinstance(focus, QLineEdit)
                    and self.handle_shortcut(event.key())):
                return True
        return super().eventFilter(obj, event)

    def keyPressEvent(self, event):
        if not self.handle_shortcut(event.key()):
            super().keyPressEvent(event)

    def handle_shortcut(self, key):
        """Run the action bound to a key, reporting whether there was one."""
        # < and > step through slides; the unshifted , and . are bound too, so
        # it works whether or not shift is held
        shortcuts = {
            Qt.Key_A: self.start_annotating_gm,
            Qt.Key_W: self.start_annotating_wm,
            Qt.Key_H: self.toggle_heatmap,
            Qt.Key_M: self.toggle_mask,
            Qt.Key_C: self.toggle_contours,
            Qt.Key_D: lambda: self.select_feature(0),
            Qt.Key_S: lambda: self.select_feature(1),
            Qt.Key_Greater: self.load_next_slide,
            Qt.Key_Period: self.load_next_slide,
            Qt.Key_Space: self.load_next_slide,
            Qt.Key_Less: self.load_previous_slide,
            Qt.Key_Comma: self.load_previous_slide,
        }
        if key == Qt.Key_Escape:
            self.toggle_is_annotating(False)
            self.canvas.current_annotation = None
            return True
        if key in shortcuts:
            shortcuts[key]()
            return True
        return False

    # -- saving annotations -------------------------------------------------

    def save_current_annotation(self):
        """Finish the in-progress ROI, prompting for a name if configured to."""
        self.current_saving_annotation = self.canvas.current_annotation
        self.toggle_is_annotating(False)

        if not self.do_ask_name:
            prefix = "GM" if self.current_saving_annotation.is_gm else "WM"
            self.confirm_save_current_annotation(
                f"{prefix}_{len(self.canvas.annotations)}")
            return

        self.ask_name = QWidget()
        self.ask_name.vlayout = QVBoxLayout()
        self.ask_name.setLayout(self.ask_name.vlayout)

        self.ask_name.name = QLineEdit("")
        self.ask_name.vlayout.addWidget(self.ask_name.name)

        self.ask_name.save = QPushButton("Save")
        self.ask_name.save.pressed.connect(self.confirm_save_current_annotation)
        self.ask_name.vlayout.addWidget(self.ask_name.save)

        self.ask_name.cancel = QPushButton("Cancel")
        self.ask_name.cancel.pressed.connect(self.cancel_save_current_annotation)
        self.ask_name.vlayout.addWidget(self.ask_name.cancel)

        self.ask_name.show()

    def confirm_save_current_annotation(self, roi_name=None):
        if roi_name is None:
            # came from the name prompt rather than an auto-generated name
            self.current_saving_annotation.name = self.ask_name.name.text()
            self.ask_name.hide()
            self.ask_name = None
        else:
            self.current_saving_annotation.name = roi_name

        self.current_saving_annotation.save()
        self.canvas.annotations.append(self.current_saving_annotation)
        self.save_annotations()

        self.current_saving_annotation = None
        self.canvas.current_annotation = None
        self.refresh_roi_list()

    def cancel_save_current_annotation(self):
        self.current_saving_annotation = None
        self.ask_name.hide()
        self.ask_name = None
        self.canvas.current_annotation = None

    # -- annotation file I/O ------------------------------------------------

    def annotation_path(self):
        """This slide's ROIs, kept inside the project rather than beside the
        archive, so the slide drive is never written to."""
        return self.project.annotation_path(self.slide_entry)

    def annotation_attributes(self):
        """Provenance stamped onto every ROI: who drew it, and when."""
        return {"author": self.project.user,
                "created": datetime.datetime.now().isoformat(timespec="seconds")}

    def save_annotations(self):
        """Write every saved ROI out as GeoJSON.

        A GM ROI is stored as its four boundary curves rather than one closed
        polygon, so downstream sampling can tell CSF from GM/WM from the walls.
        """
        attributes = self.annotation_attributes()
        annotations = []
        for a in self.canvas.annotations:
            if a.is_gm:
                annotations += [
                    a.csf_gm_seg.to_annotation(class_name="CSF", annotation_name=a.name, attributes=attributes),
                    a.right_wall.to_annotation(class_name="R", annotation_name=a.name, attributes=attributes),
                    a.gm_wm_seg.to_annotation(class_name="GM", annotation_name=a.name, attributes=attributes),
                    a.left_wall.to_annotation(class_name="L", annotation_name=a.name, attributes=attributes),
                ]
            if a.is_wm:
                annotations.append(
                    a.wm_roi.to_annotation(class_name="ROI", annotation_name=a.name, attributes=attributes))

        write_annotations(self.annotation_path(), annotations)

        # keep this slide's row in step with what is now on disk. the QC status
        # has to come along: set_row redraws the whole row, so leaving it out
        # would clear the QC fill
        self.refresh_slide_row()

    def load_annotations(self, annotations):
        """Rebuild saved ROIs from GeoJSON, grouping the parts by ROI name."""
        # first-appearance order, not a set: the ROI list is positional now,
        # so an arbitrary order would shuffle which ROI holds which number
        for name in dict.fromkeys(x.annotation_name for x in annotations):
            parts = {x.class_name: x for x in annotations
                     if x.annotation_name == name}

            annotation = Annotation(is_gm=False, is_wm=False)
            annotation.name = name
            annotation.saved = True

            # a GM ROI needs all four boundary curves to be reconstructable
            if all(k in parts for k in ("CSF", "R", "GM", "L")):
                annotation.is_gm = True
                annotation.csf_gm_seg = parts["CSF"].to_curve()
                annotation.right_wall = parts["R"].to_curve()
                annotation.gm_wm_seg = parts["GM"].to_curve()
                annotation.left_wall = parts["L"].to_curve()

            if "ROI" in parts:
                annotation.is_wm = True
                annotation.wm_roi = parts["ROI"].to_polygon()

            self.canvas.annotations.append(annotation)

        self.refresh_roi_list()
        self.canvas.update()

    # -- mouse handling -----------------------------------------------------

    def canvas_point(self, event):
        """The mouse position in canvas pixels, via Qt rather than arithmetic.

        The canvas sits inside a scroll area that is offset by the docks, the
        toolbar and the current scroll position, so `mapFromGlobal` is the only
        thing that stays correct when any of those change.
        """
        return self.canvas.mapFromGlobal(event.globalPosition().toPoint())

    def event_to_image_coords(self, event):
        """Map a mouse position into unscaled image coordinates."""
        p = self.canvas_point(event)
        return np.array([p.x(), p.y()]) / self.scale_factor

    def set_cursor_point(self, p, idx=None, poly=None):
        self.canvas.point = None if p is None else QPoint(p[0], p[1])
        self.canvas.idx = idx
        self.canvas.poly = poly
        self.canvas.update()

    def cursor_allowed(self, p):
        """Whether the cursor may be shown here.

        A GM ROI is derived by casting rays through cortex from the click, so a
        click outside GM produces a well-formed but meaningless ROI rather than
        failing. Hiding the cursor there stops that at the source. Only GM is
        restricted; deep WM is drawn corner by corner and needs the freedom.
        """
        a = self.canvas.current_annotation
        if not (self.is_annotating and a is not None and a.is_gm):
            return True
        if self.gm_bool is None:
            return True

        x, y = int(p[0]), int(p[1])
        h, w = self.gm_bool.shape
        return 0 <= y < h and 0 <= x < w and bool(self.gm_bool[y, x])

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        p = self.event_to_image_coords(event)

        # the cursor itself now shows the location, so the GMM window's marker
        # would just be a second crosshair left behind
        self.canvas.highlight = None
        self.mark_gmm_point(p)

        # while placing GM/WM seeds - and whenever snapping is off - the cursor
        # simply follows the mouse
        if not SNAP_TO_SEGMENTATION:
            self.set_cursor_point(p if self.cursor_allowed(p) else None)
            return

        a = self.canvas.current_annotation
        if a is None:
            return

        # pick which contours the cursor may snap to, based on which control
        # point is being placed next
        if a.is_gm:
            if a.csf0 is None:
                available_polys = self.tissue_polys + self.tissue_holes
            elif a.wm0 is None:
                available_polys = self.wm_polys
            elif a.csf1 is None:
                available_polys = [a.csf_poly]
            elif a.wm1 is None:
                available_polys = [a.wm_poly]
            elif self.annotation_dock.widget.annotate_adjacent_wm.isChecked():
                # the WM box corners must land inside a WM region
                if a.wm2 is None or a.wm3 is None:
                    self.snap_inside_wm(p)
                return
            else:
                return
        elif self.annotation_dock.widget.snap_to_seg.isChecked():
            if a.wm3 is None:
                self.snap_inside_wm(p)
            return
        else:
            return

        force_snap = (a.wm0 is None and a.csf0 is not None) or \
                     (a.csf1 is not None and a.wm1 is None)
        if self.annotation_dock.widget.snap_to_seg.isChecked() or force_snap:
            self.snap_to_nearest_vertex(p, available_polys)
        else:
            self.set_cursor_point(p)

    def snap_inside_wm(self, p):
        """Show the cursor only while it is inside a WM region."""
        for poly in self.wm_polys:
            if sana.geo.ray_tracing(p[0], p[1], poly):
                self.set_cursor_point(p)
                return
        self.set_cursor_point(None)

    def snap_to_nearest_vertex(self, p, available_polys):
        """Pull the cursor onto the closest vertex of the candidate contours."""
        idxs, dists = [], []
        for poly in available_polys:
            d = np.sum((p - poly) ** 2, axis=1)
            idxs.append(np.argmin(d))
            dists.append(np.min(d))

        poly_idx = np.argmin(dists)
        poly = available_polys[poly_idx]
        idx = idxs[poly_idx]
        self.set_cursor_point(poly[idx], idx=idx, poly=poly)

    # TODO: add click/drag logic for 4 control points
    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            return

        # ignore clicks that land off the image - on a dock, or the grey margin
        p = self.canvas_point(event)
        if not (0 <= p.x() < self.canvas.width()
                and 0 <= p.y() < self.canvas.height()):
            return

        # a click adds a control point, finishing the ROI once it is complete
        if self.is_annotating:
            self.canvas.select_point()
            if self.canvas.current_annotation is None or self.canvas.current_annotation.done:
                self.save_current_annotation()

    # -- slide discovery and loading ----------------------------------------

    def set_project(self, project, start_idx=0):
        """Adopt a project and show its slides, loading the first one."""
        self.project = project
        self.refresh_project_panel()

        if not project.entries:
            # an empty project is perfectly valid - the user imports next
            self.clear_slide_state()
            return

        # give the user a chance to repoint anything that moved before
        # deciding what is loadable
        if project.missing_entries():
            ResolvePathsDialog(project, self).exec()
            self.refresh_project_panel()

        # start on a slide that is actually there, so one missing archive at
        # the top of the list does not greet the user with an error
        available = [i for i, e in enumerate(project.entries)
                     if os.path.exists(e.npz_path)]
        if not available:
            self.clear_slide_state()
            return self.error_dialog(
                "None of this project's slides are reachable.\n"
                "Is the drive holding them connected?")
        self.load_slide(start_idx if start_idx in available else available[0])

    def refresh_project_panel(self):
        """Repopulate the slide list and the user readout from the project."""
        self.project_dock.widget.set_project(self.project)
        self.update_delete_button()
        self.setWindowTitle(f"SANA ROI Annotator - {self.project.name}")

    @property
    def slide_names(self):
        """Entry names, kept as a property so navigation logic is unchanged."""
        return [e.name for e in self.project.entries] if self.project else []

    def select_slide_row(self, idx):
        """Keep the list highlight on the slide being shown, without the
        selection change looping back into another load."""
        slide_list = self.project_dock.widget.slide_list
        slide_list.blockSignals(True)
        slide_list.setCurrentRow(idx)
        slide_list.blockSignals(False)

        # setCurrentRow also selects, but with signals blocked nothing told the
        # delete button, which would then contradict what the list shows
        self.update_delete_button()

    def read_slide_arrays(self, path):
        """Pull every array out of the archive in one pass, so the zip is only
        decompressed once."""
        with np.load(path, allow_pickle=True) as z:
            return {name: z[key] for name, key in self.ARRAY_KEYS.items()}

    def clear_slide_state(self):
        """Drop the current slide so a failed load cannot leave stale data up."""
        self.slide_name = None
        self.slide_tb = None
        self.features = None
        self.gm_mask = None
        self.wm_mask = None
        self.tissue_mask = None
        self.stored_gm_mask = None
        self.boundaries = {}
        self.gm_bool = self.wm_bool = None
        self.gm_polys = None
        self.wm_polys = None
        self.tissue_polys = None
        self.update_ui_actions(False)

    def load_slide(self, idx):
        if idx < 0 or idx >= len(self.slide_names):
            return self.error_dialog("Slide index out of bounds")

        entry = self.project.entries[idx]
        if not os.path.exists(entry.npz_path):
            # the drive may be unplugged, or the archive moved since import
            self.slide_idx = idx
            self.select_slide_row(idx)
            self.clear_slide_state()
            return self.error_dialog(
                f"{entry.name} is no longer where the project recorded it.\n\n"
                f"{entry.npz_path}\n\n"
                "Reconnect the drive, or remove the slide from the project.")

        try:
            self.slide_idx = idx
            self.slide_entry = entry
            self.slide_name = self.slide_entry.name

            # a [WARNING] marker beside the archive flags a suspect segmentation,
            # which plot_curves draws in red instead of green
            self.warn_user = os.path.exists(os.path.join(
                os.path.dirname(self.slide_entry.npz_path),
                f'[WARNING]_{self.slide_name}.png'))

            arrays = self.read_slide_arrays(self.slide_entry.npz_path)

            self.slide_tb = sana.image.Frame(arrays["Thumbnail"])

            # the heatmap is stored at thumbnail resolution, so one array
            # serves both the feature overlays and the re-segmenter
            self.features = sana.image.Frame(arrays["Features"])

            # masks may be saved at a coarser resolution than the thumbnail
            self.gm_mask = sana.image.Frame(arrays["GM Mask"].astype(np.uint8))
            self.wm_mask = sana.image.Frame(arrays["WM Mask"].astype(np.uint8))
            self.tissue_mask = sana.image.Frame(arrays["Tissue Mask"].astype(np.uint8))
            self.update_ui_actions(True)
            for mask in (self.tissue_mask, self.gm_mask, self.wm_mask):
                mask.resize(self.slide_tb.size())

            # the archive's own segmentation, kept as the fixed reference that
            # every Run NEUSEG result is measured against. self.gm_mask is
            # replaced by a run; this snapshot is not, so running twice reports
            # the same difference rather than converging on zero
            self.stored_gm_mask = np.asarray(
                self.gm_mask.img).squeeze().astype(bool).copy()

            # TODO: simplify polys
            self.gm_polys, self.gm_holes = self.gm_mask.to_polygons()
            self.wm_polys, self.wm_holes = self.wm_mask.to_polygons()
            self.tissue_polys, self.tissue_holes = self.tissue_mask.to_polygons()

            # tissue + WM gives a single mask with 0=background, 1=GM, 2=WM,
            # which is what the ray casting in Canvas walks
            # TODO: update this again when re-training
            self.segmentation_mask = self.tissue_mask.copy()
            self.segmentation_mask.img += self.wm_mask.img
            self.canvas.mask = self.segmentation_mask

            # interpolated contours give the ROI boundaries smooth vertices
            self.canvas.gm_polys = (
                [pdnl_sana.interpolate.interp_poly(x) for x in self.gm_polys]
                + [pdnl_sana.interpolate.interp_poly(x) for x in self.gm_holes]
            )

            # both are derived from the masks, so they are built once here
            # rather than on every redraw
            self.cache_boundaries()
            self.cache_feature_stats()

            # any previous NEUSEG run belongs to the slide we just left. the
            # plot must go with it: its points carry that slide's pixel
            # coordinates, which would land somewhere arbitrary on this one
            # both panels belong to the slide being left
            for window in (self.gmm_window, self.qc_window):
                if window is not None:
                    window.close()
            self.gmm_result = None
            self.last_gmm_pixel = None
            self.canvas.highlight = None
            self.segmentation_dock.widget.show_gmm_button.setEnabled(False)
            self.segmentation_dock.widget.status.setText("")

        except Exception as e:
            # TODO: save to a logging file/flag for re-annotation
            self.error_dialog(f"Could not load slide data...\n{e}")
            self.clear_slide_state()
            return

        self.canvas.annotations = []
        self.refresh_roi_list()
        if self.is_annotating:
            self.toggle_is_annotating()
        self.update_navigation_toolbar()
        self.select_slide_row(idx)
        self.load_rated()

        # each slide keeps its own window per channel, so this restores what
        # was set here before, or opens at full range the first time
        self.apply_feature_window(self.overlay_dock.widget.feature_combo.currentIndex())

        # kicks off the render chain: source -> overlaid -> plotted -> resized
        self.set_source_frame(self.slide_tb)
        self.update_histogram()

        if os.path.exists(self.annotation_path()):
            self.load_annotations(read_annotations(self.annotation_path()))

        # fit now, then again once Qt has laid out the dock and toolbar that
        # update_ui_actions just revealed, since those change the viewport
        self.fit_to_window()
        QTimer.singleShot(0, self.fit_to_window)

    def load_previous_slide(self):
        self.load_slide(self.slide_idx - 1)

    def load_next_slide(self):
        self.load_slide(self.slide_idx + 1)

    # -- render chain -------------------------------------------------------
    #
    # Each stage stores its result and calls the next, so any upstream change
    # (new slide, overlay toggle, zoom) repaints everything below it:
    #
    #   source -> overlaid (features) -> plotted (contours) -> resized (zoom)

    def set_source_frame(self, frame):
        self.source_frame = frame
        self.overlay_features()

    def set_overlaid_frame(self, frame):
        self.overlaid_frame = frame
        self.plot_curves()

    def set_plotted_frame(self, frame):
        self.plotted_frame = frame
        self.resize_frame()

    def set_resized_frame(self, frame):
        self.resized_frame = frame
        self.set_current_frame(frame)

    def set_current_frame(self, frame):
        self.current_frame = frame
        self.current_image = self.frame_to_qimage(frame)
        self.canvas.setPixmap(QPixmap.fromImage(self.current_image))
        self.canvas.adjustSize()

    def frame_to_qimage(self, frame):
        image_array = frame.img
        h, w = image_array.shape[:2]
        if frame.is_rgb():
            fmt = QImage.Format.Format_RGB888
        elif frame.is_binary():
            fmt = QImage.Format.Format_Mono
        else:
            fmt = QImage.Format.Format_Grayscale8

        return QImage(image_array, w, h, image_array.strides[0], fmt)

    # TODO: N/A or flag slide button (reason dropdown or text field)
    # TODO: add flip segmetnations button if the GM/WM cluseters were labeled incorrectly
    # TODO: file -> export local annotations to .zip file (save .zip in file dialog)
    # TODO: create cmap dropdown
    # TODO: flag image for re-segmentation if segmentation is bad, either draw and classify using supervised, or re-init GMM
    def feature_channel_image(self, channel):
        """One feature channel, resized to the thumbnail if it was saved coarser.

        Sized against slide_tb rather than source_frame: the stats are cached
        during load_slide, before the render chain has set a source frame.
        """
        feature = self.features.img[:, :, channel]
        h, w = self.slide_tb.img.shape[:2]
        if feature.shape[:2] != (h, w):
            feature = cv2.resize(feature, (w, h), interpolation=cv2.INTER_LINEAR)
        return feature

    def cache_feature_stats(self):
        """Per-channel tissue values and value range, refreshed on every slide.

        Background is excluded: roughly a third of the heatmap is exact zero
        outside the tissue, which would otherwise dominate both the histogram
        and the slider range.
        """
        tissue = np.squeeze(self.tissue_mask.img) > 0
        self.feature_values, self.feature_limits = {}, {}
        for _, channel in FEATURE_CHANNELS:
            values = self.feature_channel_image(channel)[tissue][::HISTOGRAM_STRIDE]
            self.feature_values[channel] = values
            self.feature_limits[channel] = (
                (float(np.nanmin(values)), float(np.nanmax(values)))
                if values.size else (0.0, 1.0))

    def feature_window(self, channel):
        """Map the 1..1000 sliders onto the real value range of this channel."""
        vmin, vmax = self.feature_limits[channel]
        widget = self.overlay_dock.widget
        lo = vmin + (widget.window.low() - 1) / 999 * (vmax - vmin)
        hi = vmin + (widget.window.high() - 1) / 999 * (vmax - vmin)
        return lo, max(hi, lo + 1e-9)

    @staticmethod
    def blend(base, colour, where, opacity):
        """Alpha blend `colour` over `base` wherever `where` is true.

        float32 rather than the default float64: this runs over every pixel on
        every slider step, and the result is rounded to uint8 anyway.
        """
        alpha = np.where(where, np.float32(opacity), np.float32(0))[:, :, None]
        return np.rint(base * (1.0 - alpha) + colour * alpha).astype(np.uint8)

    def heatmap_layer(self, base, opacity, channel, lo, hi):
        """The feature channel, clipped to the slider window and colourised.

        Blended inside the tissue only, so the background stays clean.
        """
        feature = self.feature_channel_image(channel)
        norm = np.clip((feature - lo) / (hi - lo), 0.0, 1.0)
        colour = HEATMAP_LUT[np.rint(255 * norm).astype(np.uint8)]

        return self.blend(base, colour, np.squeeze(self.tissue_mask.img) > 0, opacity)

    def mask_layer(self, base, opacity):
        """Flat GM and WM fills, in the same colours as their contours."""
        colour = np.zeros_like(base)
        colour[self.gm_bool] = MASK_COLOURS["gm"]
        colour[self.wm_bool] = MASK_COLOURS["wm"]
        return self.blend(base, colour, self.gm_bool | self.wm_bool, opacity)

    def overlay_features(self):
        """Compose the tinted frame: heatmap first, then the masks over it.

        Either layer can be off. The GM/WM contours come from the next stage of
        the render chain, so they always land on top of both.
        """
        if self.features is None:
            return

        widget = self.overlay_dock.widget
        image = self.source_frame.img

        if widget.show_heatmap.isChecked():
            channel = widget.feature_combo.currentIndex()
            lo, hi = self.feature_window(channel)
            image = self.heatmap_layer(image, widget.heatmap_alpha.slider.value() / 100,
                                       channel, lo, hi)
            # the on-canvas key names what is being shown
            self.canvas.heatmap_legend = (FEATURE_CHANNELS[channel][0],
                                          f"{lo:.4g}", f"{hi:.4g}")
        else:
            self.canvas.heatmap_legend = None

        if widget.show_mask.isChecked():
            image = self.mask_layer(image, widget.mask_alpha.slider.value() / 100)

        return self.set_overlaid_frame(
            self.source_frame if image is self.source_frame.img
            else sana.image.frame_like(self.source_frame, image))

    def update_histogram(self):
        """Redraw the histogram and the value readout for the current channel."""
        if self.features is None:
            return

        widget = self.overlay_dock.widget
        channel = widget.feature_combo.currentIndex()
        lo, hi = self.feature_window(channel)
        widget.histogram.plot(self.feature_values[channel], lo, hi,
                              FEATURE_CHANNELS[channel][0])
        widget.range_label.setText(f"{lo:.4g} to {hi:.4g}")

    def update_feature_view(self):
        """Redraw the histogram and the overlay after any control changes."""
        if self.features is None:
            return
        self.store_feature_window()
        self.update_histogram()
        self.overlay_features()

    def store_feature_window(self):
        """Remember where the window sits for whatever is on the slider now."""
        if self.window_key is not None:
            slider = self.overlay_dock.widget.window
            self.feature_windows[self.window_key] = (slider.low(), slider.high())

    def apply_feature_window(self, channel):
        """Put this slide and channel's remembered window back, without a redraw."""
        key = (self.slide_name, channel)
        low, high = self.feature_windows.get(key, (1, 1000))

        slider = self.overlay_dock.widget.window
        slider.blockSignals(True)
        slider.set_values(low, high)
        slider.blockSignals(False)
        self.window_key = key

    def select_feature_channel(self):
        """Switching channel swaps in that channel's own remembered window."""
        self.store_feature_window()
        self.apply_feature_window(self.overlay_dock.widget.feature_combo.currentIndex())
        self.update_feature_view()

    def toggle_feature_heatmap(self):
        """Menu action and checkbox are two ways to flip the same switch."""
        widget = self.overlay_dock.widget
        widget.show_heatmap.setChecked(self.show_heatmap_action.isChecked())

    def toggle_heatmap(self):
        box = self.overlay_dock.widget.show_heatmap
        box.setChecked(not box.isChecked())

    def toggle_mask(self):
        box = self.overlay_dock.widget.show_mask
        box.setChecked(not box.isChecked())

    def toggle_contours(self):
        box = self.overlay_dock.widget.show_contours
        box.setChecked(not box.isChecked())

    def select_feature(self, channel):
        """Pick a heatmap channel by key, switching the heatmap on if it is off.

        Direct selection rather than a cycle, so the keys stay meaningful if a
        third channel is ever added, and pressing one twice is a no-op.
        """
        widget = self.overlay_dock.widget
        if channel >= widget.feature_combo.count():
            return
        widget.feature_combo.setCurrentIndex(channel)
        widget.show_heatmap.setChecked(True)

    @staticmethod
    def dilate(mask, iterations=1):
        return cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8),
                          iterations=iterations).astype(bool)

    @staticmethod
    def erode(mask):
        """Shrink a mask by one pixel, so `mask & ~erode(mask)` is its rim."""
        return cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)

    def cache_boundaries(self):
        """Build the three boundary classes as pixel masks.

        GM and WM partition the tissue, so the tissue rim splits cleanly into
        the two CSF boundaries by asking which mask each rim pixel belongs to.
        Whatever is left of the GM rim is not against background, so it can
        only be against WM.

        Cached because the masks change only when a slide loads or NEUSEG runs,
        while plot_curves is called on every heatmap slider step.
        """
        tissue = np.squeeze(self.tissue_mask.img).astype(bool)
        gm = np.squeeze(self.gm_mask.img).astype(bool)
        wm = np.squeeze(self.wm_mask.img).astype(bool)

        # the whole derivation rests on this, so check it rather than assume it
        assert np.array_equal(gm | wm, tissue), (
            "gm_mask and wm_mask must partition tissue_mask: "
            f"{int((gm & wm).sum())} px in both, "
            f"{int((tissue & ~gm & ~wm).sum())} tissue px in neither, "
            f"{int(((gm | wm) & ~tissue).sum())} px outside the tissue")

        # kept for the cursor test and the mask overlay, both of which run
        # far more often than the masks change
        self.gm_bool, self.wm_bool = gm, wm

        edge = tissue & ~self.erode(tissue)              # tissue against background
        lines = {"gm_csf": edge & gm,                    # ...the GM part of it
                 "wm_csf": edge & wm,                    # ...the WM part of it
                 "gm_wm": (gm & ~self.erode(gm)) & ~edge}  # GM rim facing WM

        # grown to the drawn width once here, not on every redraw
        self.boundaries = {k: self.dilate(v, CONTOUR_WIDTH // 2)
                           for k, v in lines.items()}

    def plot_curves(self):
        """Paint the boundary classes onto the frame, GM/WM last so it wins."""
        # passing the frame straight through also skips an 18 MB copy
        if not self.overlay_dock.widget.show_contours.isChecked():
            return self.set_plotted_frame(self.overlaid_frame)

        img = self.overlaid_frame.img.copy()
        for name in ("gm_csf", "wm_csf", "gm_wm"):
            colour = CONTOUR_WARNING_COLOUR if self.warn_user else CONTOUR_COLOURS[name]
            img[self.boundaries[name]] = colour

        self.set_plotted_frame(sana.image.frame_like(self.source_frame, img))

    def resize_frame(self):
        """Re-render the frame at the current zoom level."""
        w = int(round(self.scale_factor * self.plotted_frame.size()[0]))
        h = int(round(self.scale_factor * self.plotted_frame.size()[1]))

        resized_frame = self.plotted_frame.copy()
        resized_frame.resize(sana.geo.Point(w, h), interpolation=cv2.INTER_LINEAR)

        # the canvas scales annotation coordinates by the same factor
        self.canvas.scale_factor = self.scale_factor
        self.set_resized_frame(resized_frame)

    # -- zoom and window sizing ---------------------------------------------

    def set_scale_factor(self, scale_factor):
        self.scale_factor = scale_factor
        self.resize_frame()

    def zoom_in(self):
        self.set_scale_factor(self.scale_factor * ZOOM_STEP)
        self.update_scroll_bars(ZOOM_STEP)

        too_big = self.current_image.width() > 10000 or self.current_image.height() > 10000
        self.zoom_in_action.setEnabled(not too_big)
        self.zoom_out_action.setEnabled(True)

    def zoom_out(self):
        self.set_scale_factor(self.scale_factor / ZOOM_STEP)
        self.update_scroll_bars(1 / ZOOM_STEP)

        too_small = self.current_image.width() < 100 or self.current_image.height() < 100
        self.zoom_out_action.setEnabled(not too_small)
        self.zoom_in_action.setEnabled(True)

    def reset_zoom(self):
        self.set_scale_factor(1.0)
        self.update_scroll_bars(1)
        self.zoom_in_action.setEnabled(True)
        self.zoom_out_action.setEnabled(True)

    def update_scroll_bars(self, factor):
        for bar in (self.scroll_area.horizontalScrollBar(),
                    self.scroll_area.verticalScrollBar()):
            bar.setValue(int(factor * bar.value()
                             + ((factor - 1) * bar.pageStep() / 2)))

    def fit_to_window(self):
        """Scale the slide so the whole thumbnail fits the current window.

        Only the zoom level changes - the window is never moved or resized, so
        loading a new slide leaves it exactly where the user put it.
        `maximumViewportSize` reports the space available with no scroll bars
        showing, which is what the image will need once it fits.
        """
        if self.source_frame is None:
            return

        viewport = self.scroll_area.maximumViewportSize()
        slide_w, slide_h = self.source_frame.size()[:2]
        if viewport.width() < 1 or viewport.height() < 1 or slide_w < 1 or slide_h < 1:
            return

        self.set_scale_factor(min(viewport.width() / slide_w,
                                  viewport.height() / slide_h))
        self.zoom_in_action.setEnabled(True)
        self.zoom_out_action.setEnabled(True)

    # -- docks, toolbar, menus ----------------------------------------------

    def init_docks(self):
        """Build the three right-hand panels and wire their controls up."""
        # the project panel lives on the left and is always visible, since it
        # is how slides are imported and chosen
        self.project_dock = self.add_dock("Project", ProjectWidget,
                                          area=Qt.DockWidgetArea.LeftDockWidgetArea)
        self.project_dock.show()

        # top to bottom in the order the work is done: annotate, adjust what
        # you can see, recompute, then record a verdict
        self.annotation_dock = self.add_dock("1 - Annotate", AnnotationWidget)
        self.overlay_dock = self.add_dock("2 - Display", OverlayWidget)
        self.segmentation_dock = self.add_dock("3 - Segmentation", SegmentationWidget)

        # stack them rather than letting Qt tab them together
        for above, below in ((self.annotation_dock, self.overlay_dock),
                             (self.overlay_dock, self.segmentation_dock)):
            self.splitDockWidget(above, below, Qt.Vertical)

        panel = self.project_dock.widget
        panel.add_button.pressed.connect(self.add_slides)
        panel.delete_button.pressed.connect(self.delete_selected_slides)
        panel.slide_list.itemSelectionChanged.connect(self.update_delete_button)
        panel.change_user_button.pressed.connect(self.change_user)
        panel.slide_list.currentRowChanged.connect(self.load_slide)
        panel.slide_list.paths_dropped.connect(self.import_paths)

        overlays = self.overlay_dock.widget
        overlays.show_heatmap.stateChanged.connect(self.update_feature_view)
        overlays.feature_combo.currentIndexChanged.connect(self.select_feature_channel)
        overlays.window.valueChanged.connect(self.update_feature_view)
        overlays.heatmap_alpha.slider.valueChanged.connect(self.overlay_features)
        overlays.show_contours.stateChanged.connect(self.plot_curves)
        overlays.show_mask.stateChanged.connect(self.overlay_features)
        overlays.mask_alpha.slider.valueChanged.connect(self.overlay_features)

        annotation = self.annotation_dock.widget
        annotation.annotate_gm_button.pressed.connect(self.start_annotating_gm)
        annotation.annotate_wm_button.pressed.connect(self.start_annotating_wm)
        annotation.snap_to_seg.stateChanged.connect(self.plot_curves)
        annotation.roi_list.itemSelectionChanged.connect(self.select_rois)
        annotation.delete_roi_button.pressed.connect(self.delete_selected_rois)

        segmentation = self.segmentation_dock.widget
        segmentation.run_button.pressed.connect(self.run_neuseg)
        segmentation.show_gmm_button.pressed.connect(self.show_gmm_window)

    def add_dock(self, name, widget_class, area=Qt.DockWidgetArea.RightDockWidgetArea):
        dock = DockWidget(name, widget_class)
        dock.hide()
        dock.setAllowedAreas(area)
        self.addDockWidget(area, dock)
        return dock

    def init_navigation(self):
        self.navigation_toolbar = QToolBar("Slide Navigation")
        self.navigation_toolbar.hide()
        self.navigation_toolbar.setFloatable(False)
        self.navigation_toolbar.setMovable(False)
        self.addToolBar(self.navigation_toolbar)

        self.previous_button = QPushButton("Previous Slide")
        self.previous_button.pressed.connect(self.load_previous_slide)
        self.navigation_toolbar.addWidget(self.previous_button)

        self.current_label = QLabel("")
        self.current_label.setAlignment(
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignHCenter)
        self.navigation_toolbar.addWidget(self.current_label)

        self.next_button = QPushButton("Next Slide")
        self.next_button.pressed.connect(self.load_next_slide)
        self.navigation_toolbar.addWidget(self.next_button)

        # an expanding spacer pushes everything below to the far right
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.navigation_toolbar.addWidget(spacer)

        # accented, because it is the one toolbar control that writes anything
        self.qc_button = QPushButton("QC Check")
        self.qc_button.pressed.connect(self.show_qc_window)
        self.qc_button.setStyleSheet(
            "QPushButton { background-color: %s; color: white; font-weight: bold;"
            " padding: 4px 14px; border: none; border-radius: 4px; }"
            "QPushButton:hover { background-color: #1565c0; }"
            "QPushButton:disabled { background-color: #b0b0b0; }" % RANGE_SLIDER_COLOUR)
        self.navigation_toolbar.addWidget(self.qc_button)

        self.has_ratings = QCheckBox("Has Ratings")
        self.has_ratings.stateChanged.connect(self.save_rated)
        self.navigation_toolbar.addWidget(self.has_ratings)

        self.contour_legend = QLabel("   " + CONTOUR_LEGEND)
        self.navigation_toolbar.addWidget(self.contour_legend)

        self.shortcut_label = QLabel(SHORTCUT_LEGEND)
        self.shortcut_label.setStyleSheet("color: gray;")
        self.navigation_toolbar.addWidget(self.shortcut_label)

    def update_navigation_toolbar(self):
        """Show the slide name and grey out navigation at either end."""
        self.current_label.setText(self.slide_name)
        self.current_label.setFont(QFont("Arial", SLIDE_NAME_POINT_SIZE))

        has_previous = self.slide_idx != 0
        self.previous_button.setEnabled(has_previous)
        self.previous_slide_action.setEnabled(has_previous)

        has_next = self.slide_idx != len(self.slide_names) - 1
        self.next_button.setEnabled(has_next)
        self.next_slide_action.setEnabled(has_next)

    def create_actions(self):
        self.new_project_action = QAction("New Project...", self, shortcut="Ctrl+N", triggered=self.action_new_project)
        self.open_project_action = QAction("Open Project...", self, shortcut="Ctrl+O", triggered=self.action_open_project)
        self.add_slides_action = QAction("Add Slides...", self, shortcut="Ctrl+I", triggered=self.add_slides)
        self.previous_slide_action = QAction("Open Previous Slide", self, shortcut="Ctrl+<", triggered=self.load_previous_slide, enabled=False)
        self.next_slide_action = QAction("Open Next Slide", self, shortcut="Ctrl+>", triggered=self.load_next_slide, enabled=False)
        self.exit_action = QAction("Exit", self, shortcut="Ctrl+Q", triggered=self.close)
        self.reset_action = QAction("Reset Annotations", self, shortcut="Ctrl+R", triggered=self.reset_annotations, enabled=False)
        self.zoom_in_action = QAction("Zoom In (25%)", self, shortcut="Ctrl+=", enabled=False, triggered=self.zoom_in)
        self.zoom_out_action = QAction("Zoom Out (25%)", self, shortcut="Ctrl+-", enabled=False, triggered=self.zoom_out)
        self.reset_zoom_action = QAction("Reset Zoom", self, shortcut="Ctrl+0", enabled=False, triggered=self.reset_zoom)
        self.fit_action = QAction("Fit to Window", self, shortcut="Ctrl+F", enabled=False, triggered=self.fit_to_window)
        self.show_heatmap_action = QAction("Show Feature Heatmap", self, shortcut="Ctrl+1", enabled=False, checkable=True, triggered=self.toggle_feature_heatmap)

        # Qt maps Ctrl to Command on macOS. Both the bare and the shifted key
        # are bound so Cmd+= and Cmd++ (i.e. Cmd+Shift+=) both zoom in.
        self.zoom_in_action.setShortcuts(["Ctrl+=", "Ctrl++"])
        self.zoom_out_action.setShortcuts(["Ctrl+-", "Ctrl+_"])

    def create_menus(self):
        self.file_menu = QMenu("File", self)
        self.file_menu.addAction(self.new_project_action)
        self.file_menu.addAction(self.open_project_action)
        self.file_menu.addSeparator()
        self.file_menu.addAction(self.add_slides_action)
        self.file_menu.addSeparator()
        self.file_menu.addAction(self.previous_slide_action)
        self.file_menu.addAction(self.next_slide_action)
        self.file_menu.addSeparator()
        self.file_menu.addAction(self.exit_action)

        self.edit_menu = QMenu("Edit", self)
        self.edit_menu.addAction(self.reset_action)

        self.view_menu = QMenu("View", self)
        self.view_menu.addAction(self.show_heatmap_action)
        self.view_menu.addAction(self.overlay_dock.toggleViewAction())

        self.window_menu = QMenu("Window", self)
        self.window_menu.addAction(self.zoom_in_action)
        self.window_menu.addAction(self.zoom_out_action)
        self.window_menu.addAction(self.reset_zoom_action)
        self.window_menu.addSeparator()
        self.window_menu.addAction(self.fit_action)

        for menu in (self.file_menu, self.edit_menu, self.view_menu, self.window_menu):
            self.menuBar().addMenu(menu)

    def update_ui_actions(self, flag):
        """Enable the slide-dependent actions once a slide is loaded."""
        self.scroll_area.setVisible(flag)
        for action in (self.zoom_in_action, self.zoom_out_action,
                       self.reset_zoom_action, self.fit_action,
                       self.show_heatmap_action, self.reset_action):
            action.setEnabled(flag)

        # the slide-dependent panels come and go with the slide; the project
        # panel stays put
        self.annotation_dock.setVisible(flag)
        self.overlay_dock.setVisible(flag)
        self.segmentation_dock.setVisible(flag)
        self.navigation_toolbar.setVisible(flag)

    # -- actions ------------------------------------------------------------

    # -- projects -----------------------------------------------------------

    def choose_project(self):
        """Startup prompt: begin a project, or pick up an existing one."""
        box = QMessageBox(self)
        box.setWindowTitle("SANA ROI Annotator")
        box.setText("Create a new project, or open an existing one?")
        new_button = box.addButton("New Project...", QMessageBox.AcceptRole)
        open_button = box.addButton("Open Project...", QMessageBox.AcceptRole)
        box.addButton("Cancel", QMessageBox.RejectRole)
        box.exec()

        if box.clickedButton() is new_button:
            return self.new_project()
        if box.clickedButton() is open_button:
            return self.open_project()
        return None

    def new_project(self):
        """Pick a folder, then ask for the user name - the only time it is asked."""
        path = QFileDialog.getExistingDirectory(
            self, caption="Choose a folder for the new project")
        if not path:
            return None
        if prj.Project.is_project(path):
            self.error_dialog("That folder already holds a project.\n"
                              "Use Open Project instead.")
            return None

        user, ok = QInputDialog.getText(self, "New Project",
                                        "User name (Marked in Annotation):")
        if not ok or not user.strip():
            return None

        # the name is written into the manifest, so reopening never re-asks
        return prj.Project.create(path, user.strip())

    def open_project(self):
        path = QFileDialog.getExistingDirectory(self, caption="Open Project")
        if not path:
            return None
        try:
            return prj.Project.open(path)
        except Exception as e:
            self.error_dialog(f"Could not open project...\n{e}")
            return None

    def action_new_project(self):
        project = self.new_project()
        if project is not None:
            self.set_project(project)

    def action_open_project(self):
        project = self.open_project()
        if project is not None:
            self.set_project(project)

    def change_user(self):
        """Let whoever is sitting down claim their own ROIs."""
        user, ok = QInputDialog.getText(self, "Change User", "User name:",
                                        text=self.project.user)
        if ok and user.strip():
            self.project.set_user(user.strip())
            self.refresh_project_panel()

    # -- ROIs on this slide -------------------------------------------------

    def refresh_roi_list(self):
        """Rebuild the ROI list from the canvas, keeping the numbering visible."""
        roi_list = self.annotation_dock.widget.roi_list
        roi_list.blockSignals(True)
        roi_list.clear()
        for a in self.canvas.annotations:
            kind = "GM" if a.is_gm else "WM"
            roi_list.addItem(f"{a.name}    ({kind})")
        roi_list.blockSignals(False)

        self.canvas.selected = set()
        self.update_delete_roi_button()

    def update_delete_roi_button(self):
        widget = self.annotation_dock.widget
        widget.delete_roi_button.setEnabled(bool(widget.roi_list.selectedIndexes()))

    def select_rois(self):
        """Highlight the ROIs picked in the list, on the slide."""
        widget = self.annotation_dock.widget
        self.canvas.selected = {i.row() for i in widget.roi_list.selectedIndexes()}
        self.update_delete_roi_button()
        self.canvas.update()

    def renumber_rois(self):
        """Re-apply sequential names so a delete leaves no gap in the numbering.

        The index is the position in the list, matching how names are handed
        out in the first place. Hand-typed names are skipped.
        """
        for i, a in enumerate(self.canvas.annotations):
            if AUTO_ROI_NAME.match(a.name):
                a.name = f"{'GM' if a.is_gm else 'WM'}_{i}"

    def delete_selected_rois(self):
        """Remove the picked ROIs, renumber the rest, and rewrite the file."""
        widget = self.annotation_dock.widget
        rows = {i.row() for i in widget.roi_list.selectedIndexes()}
        if not rows:
            return

        self.canvas.annotations = [a for i, a in enumerate(self.canvas.annotations)
                                   if i not in rows]
        self.renumber_rois()
        self.save_annotations()      # also refreshes the slide's row
        self.refresh_roi_list()
        self.canvas.update()

    # -- ratings ------------------------------------------------------------

    def load_rated(self):
        """Show this slide's flag without writing it straight back."""
        self.has_ratings.blockSignals(True)
        self.has_ratings.setChecked(self.project.is_rated(self.slide_name))
        self.has_ratings.blockSignals(False)

    def save_rated(self):
        """Record the flag in the manifest and recolour the row."""
        if self.project is None or self.slide_name is None:
            return
        self.project.set_rated(self.slide_name, self.has_ratings.isChecked())
        self.refresh_slide_row()

    # -- quality control ----------------------------------------------------

    def show_qc_window(self):
        """Open the QC panel for the current slide, prefilled from the sheet."""
        if self.project is None or self.slide_name is None:
            return self.error_dialog("Load a slide first.")

        if self.qc_window is None:
            self.qc_window = QCWindow(self)
            self.qc_window.saved.connect(self.save_qc)

        self.qc_window.set_state(self.slide_name,
                                 self.project.qc_record(self.slide_name))
        self.qc_window.show()
        self.qc_window.raise_()

    def save_qc(self):
        """Record the verdict and rewrite the sheet, then recolour the row."""
        if self.qc_window is None or self.project is None or self.slide_name is None:
            return

        self.project.set_qc(self.slide_name, self.qc_window.status(),
                            self.qc_window.bad_tissue.isChecked(),
                            self.qc_window.comments.text().strip())
        self.refresh_slide_row()

    def refresh_slide_row(self):
        """Redraw just the current slide's row, with every signal it carries.

        Only one row can have changed, so this avoids rebuilding the list - but
        it must pass all three signals, since set_row redraws the row entirely.
        """
        if self.project is None or self.slide_entry is None:
            return
        self.project_dock.widget.set_row(
            self.slide_idx, self.slide_entry,
            self.project.annotation_count(self.slide_entry),
            self.project.qc_record(self.slide_name),
            self.project.is_rated(self.slide_name))
        self.project_dock.widget.update_summary(self.project)

    # -- removing slides ----------------------------------------------------

    def update_delete_button(self):
        panel = self.project_dock.widget
        panel.delete_button.setEnabled(bool(panel.slide_list.selectedIndexes()))

    def delete_selected_slides(self):
        """Drop the selected slides, after confirming what will be removed."""
        panel = self.project_dock.widget
        rows = sorted(i.row() for i in panel.slide_list.selectedIndexes())
        entries = [self.project.entries[r] for r in rows]
        if not entries:
            return

        listed = "\n".join("  " + e.name for e in entries[:8])
        more = f"\n  ...and {len(entries) - 8} more" if len(entries) > 8 else ""
        box = QMessageBox(self)
        box.setWindowTitle("Delete slides")
        box.setText(f"Remove {len(entries)} slide(s) from the project?\n\n{listed}{more}")
        box.setInformativeText("Their annotations and QC rows are deleted.\n"
                               "The .npz archives themselves are left untouched.")
        box.setStandardButtons(QMessageBox.Yes | QMessageBox.Cancel)
        box.setDefaultButton(QMessageBox.Cancel)
        if box.exec() != QMessageBox.Yes:
            return

        self.project.remove_entries(entries)
        self.refresh_project_panel()

        # the slide on screen may have just been removed
        if not self.project.entries:
            self.clear_slide_state()
        else:
            self.load_slide(min(rows[0], len(self.project.entries) - 1))

    def add_slides(self):
        """Import through the file chooser; dropping files does the same thing."""
        paths, _ = QFileDialog.getOpenFileNames(
            self, caption="Import slide archives",
            filter="Slide data (*.npz *.pkl)")
        if paths:
            self.import_paths(paths)

    def import_paths(self, paths):
        """Add chosen or dropped paths, then say what could not be imported."""
        if self.project is None:
            return self.error_dialog("Create or open a project first.")

        added, rejected = self.project.add_paths(paths)
        self.refresh_project_panel()

        if rejected:
            shown = "\n".join(f"{os.path.basename(p)}  -  {why}"
                              for p, why in rejected[:10])
            more = f"\n...and {len(rejected) - 10} more" if len(rejected) > 10 else ""
            self.info_dialog(f"Skipped {len(rejected)} file(s):\n\n{shown}{more}")

        # jump straight into the first slide if nothing was loaded yet
        if added and self.slide_name is None:
            self.load_slide(self.project.entries.index(added[0]))

    def reset_annotations(self):
        self.canvas.reset_annotations()
        self.save_annotations()
        self.refresh_roi_list()

    def start_annotating_gm(self):
        self.canvas.current_annotation = Annotation(
            is_gm=True,
            is_wm=self.annotation_dock.widget.annotate_adjacent_wm.isChecked())
        self.canvas.gm_width = self.annotation_dock.widget.gm_width_spinbox.value()
        self.toggle_is_annotating(True)

    def start_annotating_wm(self):
        self.canvas.current_annotation = Annotation(is_gm=False, is_wm=True)
        self.toggle_is_annotating(True)

    # -- running NEUSEG -----------------------------------------------------

    def run_neuseg(self):
        """Recompute the GM/WM segmentation for this slide, off the GUI thread.

        The pipeline reads its scaling parameters from the slide's _log.pkl, so
        a slide imported without one cannot be re-run.
        """
        panel = self.segmentation_dock.widget
        if self.slide_entry is None or self.features is None:
            return self.error_dialog("Load a slide first.")
        if not (self.slide_entry.log_path
                and os.path.exists(self.slide_entry.log_path)):
            return self.error_dialog(
                "This slide has no _log.pkl, which NEUSEG needs for its scaling "
                "parameters.\nImport the log beside the .npz and try again.")

        panel.run_button.setEnabled(False)
        panel.progress.setValue(0)
        panel.progress.show()

        # the bar is driven between stage reports, and timed so the next run
        # can predict itself from this one
        self.run_start = self.stage_start = time.time()
        self.stage_text = "Starting..."
        self.stage_from, self.stage_to = 0, NeusegWorker.STAGES[0][0]
        panel.status.setText(self.stage_text)
        self.progress_timer.start(PROGRESS_TICK_MS)

        self.neuseg_worker = NeusegWorker(self.features, self.tissue_mask,
                                          self.slide_tb, self.slide_entry.log_path)
        self.neuseg_worker.progress.connect(self.on_neuseg_progress)
        self.neuseg_worker.finished_ok.connect(self.on_neuseg_finished)
        self.neuseg_worker.failed.connect(self.on_neuseg_failed)
        self.neuseg_worker.start()

    def on_neuseg_progress(self, percent, stage):
        """A real stage boundary: snap to it and re-anchor the interpolation."""
        panel = self.segmentation_dock.widget
        panel.progress.setValue(percent)
        panel.status.setText(stage)

        self.stage_text = stage
        self.stage_start = time.time()
        self.stage_from = percent
        # the next stage's start is where this one may creep up to
        starts = [start for start, _ in NeusegWorker.STAGES] + [100]
        following = [v for v in starts if v > percent]
        self.stage_to = following[0] if following else 100

    def tick_neuseg_progress(self):
        """Move the bar between stage reports, and show the time left.

        The GMM fit gives no sub-step feedback, so the position inside a stage
        is inferred from elapsed time. It eases toward the next stage's start
        without reaching it, so a slow run never looks finished early.
        """
        panel = self.segmentation_dock.widget
        elapsed = time.time() - self.run_start
        expected = max(self.expected_run_seconds, 1.0)

        # the bar tracks the clock, so it agrees with the time left; the stage
        # bounds keep an overrunning estimate from reaching the next stage
        target = 100 * min(elapsed / expected, 1.0)
        panel.progress.setValue(
            int(min(max(target, self.stage_from), self.stage_to - 1)))

        # the countdown goes in the status label, not on the bar: the native
        # macOS progress bar draws no text, so setFormat is invisible there
        remaining = expected - elapsed
        left = (f"about {int(remaining) // 60}:{int(remaining) % 60:02d} left"
                if remaining >= 1 else "almost done")
        panel.status.setText(f"{self.stage_text}   -   {left}")

    def stop_neuseg_progress(self):
        self.progress_timer.stop()
        self.segmentation_dock.widget.progress.hide()

    def on_neuseg_failed(self, message):
        self.stop_neuseg_progress()
        panel = self.segmentation_dock.widget
        panel.run_button.setEnabled(True)
        panel.status.setText("Failed")
        self.error_dialog(f"NEUSEG failed...\n{message}")

    def on_neuseg_finished(self, result):
        """Adopt the new masks for this session and report what changed.

        Nothing is written: re-running unchanged code reproduces what the
        archive already holds, so the point of the run is the GMM plot and the
        difference report, not a new file.
        """
        # this run is the best estimate for the next one
        self.expected_run_seconds = time.time() - self.run_start
        self.stop_neuseg_progress()

        panel = self.segmentation_dock.widget
        panel.run_button.setEnabled(True)

        # measured against the archive, never against the previous run
        before = self.stored_gm_mask
        after = np.asarray(result["gm_mask"]).squeeze().astype(bool)
        changed = int((before != after).sum()) if before.shape == after.shape else -1
        if changed < 0:
            summary = "Stored segmentation has a different shape"
        elif changed == 0:
            summary = "Matches the stored segmentation exactly"
        else:
            plural = "pixel differs" if changed == 1 else "pixels differ"
            summary = f"{changed:,} {plural} from the stored segmentation"
        panel.status.setText(summary)

        # show the new result on the slide
        self.gm_mask = sana.image.frame_like(self.tissue_mask, after.astype(np.uint8))
        self.wm_mask = sana.image.frame_like(
            self.tissue_mask, np.asarray(result["wm_mask"]).squeeze().astype(np.uint8))
        # the masks changed, so the boundaries drawn from them must be rebuilt
        self.cache_boundaries()

        # these feed ROI snapping, not the drawing
        self.gm_polys, self.gm_holes = self.gm_mask.to_polygons()
        self.wm_polys, self.wm_holes = self.wm_mask.to_polygons()
        self.plot_curves()

        self.gmm_result = result
        panel.show_gmm_button.setEnabled(True)
        self.show_gmm_window()

    # -- the GMM viewer -----------------------------------------------------

    def show_gmm_window(self):
        """Plot a sample of the tissue pixels the GMM was scored on."""
        if self.gmm_result is None:
            return

        tissue = self.gmm_result["tissue"]
        rows, cols = np.nonzero(tissue)
        density = self.feature_channel_image(0)[tissue]
        size = self.feature_channel_image(1)[tissue]
        gm_prob = self.gmm_result["gm_prob"][tissue]

        # a fixed seed keeps the same pixels on screen between openings
        rng = np.random.default_rng(0)
        n = min(SCATTER_SAMPLE, rows.size)
        pick = rng.choice(rows.size, size=n, replace=False)

        if self.gmm_window is None:
            self.gmm_window = GMMWindow(self)
            self.gmm_window.hovered.connect(self.highlight_feature_pixel)
            # deferred: the window still reports itself visible inside closeEvent
            self.gmm_window.closed.connect(
                lambda: QTimer.singleShot(0, self.update_mouse_tracking))
        self.gmm_window.set_data(
            density[pick], size[pick], gm_prob[pick],
            np.stack([rows[pick], cols[pick]], axis=1),
            f"{self.slide_name}  -  {n:,} of {rows.size:,} tissue pixels")
        self.gmm_window.show()
        self.gmm_window.raise_()

        # hovering the slide only reaches mouseMoveEvent while tracking is on
        self.update_mouse_tracking()

    def mark_gmm_point(self, p):
        """The reverse of the hover above: moving over the slide rings the point
        in the GMM plot that this pixel contributed.

        Skipped unless the plot is actually open, and only redrawn when the
        pixel under the mouse changes, so dragging across the slide does not
        queue a redraw per mouse event.
        """
        if self.gmm_result is None or self.gmm_window is None:
            return
        if not self.gmm_window.isVisible():
            return

        # the slide is in thumbnail pixels; the GMM was fit on the heatmap grid
        h, w = self.slide_tb.img.shape[:2]
        fh, fw = self.features.img.shape[:2]
        col = int(p[0] * fw / w)
        row = int(p[1] * fh / h)
        if not (0 <= row < fh and 0 <= col < fw):
            return
        if (row, col) == self.last_gmm_pixel:
            return
        self.last_gmm_pixel = (row, col)

        self.gmm_window.mark_feature(float(self.features.img[row, col, 0]),
                                     float(self.features.img[row, col, 1]))

    def highlight_feature_pixel(self, rowcol):
        """Mark on the slide where a hovered scatter point was measured."""
        # a hover that outlives its slide would point at the wrong place
        if self.gmm_result is None:
            return
        if rowcol is None:
            self.canvas.highlight = None
        else:
            row, col = rowcol
            # the heatmap may be coarser than the thumbnail, so rescale
            h, w = self.slide_tb.img.shape[:2]
            fh, fw = self.features.img.shape[:2]
            self.canvas.highlight = QPoint(int(col * w / fw), int(row * h / fh))
        self.canvas.update()


# ---------------------------------------------------------------------------
# 4. Qt widgets
# ---------------------------------------------------------------------------

class SlideListWidget(QListWidget):
    """The project's slide list, which also accepts files dropped onto it."""

    paths_dropped = Signal(list)

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        # only take file drops; anything else is left to Qt
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    # dragging over the list is accepted on the same terms as entering it
    dragMoveEvent = dragEnterEvent

    def dropEvent(self, event):
        paths = [u.toLocalFile() for u in event.mimeData().urls() if u.isLocalFile()]
        if paths:
            self.paths_dropped.emit(paths)
            event.acceptProposedAction()


class ProjectWidget(QWidget):
    """Left-hand panel: who is annotating, and which slides the project holds."""

    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # the user is fixed per project so it is never retyped, but it is shown
        # here so it is obvious who the ROIs will be attributed to
        self.user_layout = QHBoxLayout()
        self.layout.addLayout(self.user_layout)
        self.user_label = QLabel("User: -")
        self.user_layout.addWidget(self.user_label)
        self.change_user_button = QPushButton("Change")
        self.user_layout.addWidget(self.change_user_button)

        self.add_button = QPushButton("Add Slides...")
        self.layout.addWidget(self.add_button)

        # a tally of everything that needs attention, in one line
        self.summary = QLabel("")
        self.summary.setToolTip("\n".join(f"{glyph}  {label}"
                                           for _, glyph, _, label in SUMMARY_PARTS))
        self.layout.addWidget(self.summary)

        # extended selection gives ctrl+A, shift-click and drag for free
        self.slide_list = SlideListWidget()
        self.slide_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.layout.addWidget(self.slide_list)

        self.hint = QLabel("...or drag .npz / _log.pkl files or folders here")
        self.hint.setWordWrap(True)
        self.layout.addWidget(self.hint)

        # enabled only while something is selected
        self.delete_button = QPushButton("Delete Selected")
        self.delete_button.setEnabled(False)
        self.layout.addWidget(self.delete_button)

    def set_project(self, project):
        """Rebuild the list from the project, one colour-coded row per slide.

        Signals are blocked while rebuilding so repopulating the list does not
        look like the user picking a different slide.
        """
        self.user_label.setText(f"User: {project.user or '-'}")

        self.slide_list.blockSignals(True)
        self.slide_list.clear()
        for entry in project.entries:
            item = QListWidgetItem()
            self.slide_list.addItem(item)
            self.set_row(self.slide_list.count() - 1, entry,
                         project.annotation_count(entry),
                         project.qc_record(entry.name),
                         project.is_rated(entry.name))
        self.slide_list.blockSignals(False)
        self.update_summary(project)

    def update_summary(self, project):
        """Count the slides and everything flagged on them, nonzero parts only."""
        counts = dict.fromkeys((part[0] for part in SUMMARY_PARTS), 0)
        for entry in project.entries:
            if entry.status == prj.STATUS_NO_LOG:
                counts["no_log"] += 1
            elif entry.status == prj.STATUS_MISSING:
                counts["missing"] += 1

            record = project.qc_record(entry.name)
            if record["status"] == "Minor errors":
                counts["minor"] += 1
            elif record["status"] == "Fail":
                counts["fail"] += 1
            if record["bad_tissue"]:
                counts["bad"] += 1
            if project.is_rated(entry.name):
                counts["rated"] += 1

        flags = "".join(
            f'&nbsp; <span style="color:{colour}">{glyph}</span>{counts[key]}'
            if colour else f"&nbsp; {glyph}{counts[key]}"
            for key, glyph, colour, _ in SUMMARY_PARTS if counts[key])
        self.summary.setText(f"<b>{len(project.entries)}</b> slides{flags}")

    def set_row(self, idx, entry, n_annotations, qc=prj.QC_EMPTY, rated=False):
        """Fill in one row. Kept separate so a single slide can be refreshed
        after it is annotated, rather than re-reading every project file."""
        item = self.slide_list.item(idx)
        if item is None:
            return
        text, icon, tip, struck = entry_row(entry, n_annotations, qc, rated)
        item.setText(text)
        item.setToolTip(tip)
        # an empty icon clears the chip when a row stops being flagged
        item.setIcon(icon)

        # bad tissue: struck through, and red because Qt draws the strike in
        # the text colour. An empty brush restores the theme default.
        font = item.font()
        font.setStrikeOut(struck)
        item.setFont(font)
        item.setForeground(QColor(QC_BAD_TISSUE_COLOUR) if struck else QBrush())


class ResolvePathsDialog(QDialog):
    """Repoint slides whose archives are no longer where they were imported from.

    Locating one usually fixes the rest: a drive remounted under another name,
    or a cohort folder moved, changes the same leading path for every slide.
    """

    def __init__(self, project, parent=None):
        super().__init__(parent)
        self.project = project
        self.setWindowTitle("Missing slides")
        self.resize(620, 340)

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.summary = QLabel("")
        self.summary.setWordWrap(True)
        self.layout.addWidget(self.summary)

        self.missing_list = QListWidget()
        self.layout.addWidget(self.missing_list)

        buttons = QHBoxLayout()
        self.layout.addLayout(buttons)

        self.locate_button = QPushButton("Locate Selected...")
        self.locate_button.pressed.connect(self.locate)
        buttons.addWidget(self.locate_button)
        buttons.addStretch()

        self.open_button = QPushButton("Open Project")
        self.open_button.pressed.connect(self.accept)
        buttons.addWidget(self.open_button)

        self.refresh()

    def refresh(self, note=""):
        missing = self.project.missing_entries()
        self.missing_list.clear()
        for entry in missing:
            self.missing_list.addItem(f"{entry.name}\n      {entry.npz_path}")

        if missing:
            self.missing_list.setCurrentRow(0)
            self.summary.setText(
                f"{note}{len(missing)} slide(s) are not where the project recorded "
                "them. Locate one and any others that moved the same way are "
                "repointed with it. Whatever is left unresolved stays in the "
                "project, marked in the slide list.")
        else:
            self.summary.setText(f"{note}All slides found.")

        self.locate_button.setEnabled(bool(missing))
        self.open_button.setText("Open Project" if not missing else "Open Anyway")
        return missing

    def locate(self):
        """Point one slide at its archive, then follow the same move for the rest."""
        missing = self.project.missing_entries()
        row = max(self.missing_list.currentRow(), 0)
        if row >= len(missing):
            return
        entry = missing[row]

        found, _ = QFileDialog.getOpenFileName(
            self, f"Locate {entry.name}", os.path.dirname(entry.npz_path),
            "Slide archive (*.npz)")
        if not found:
            return

        old_prefix, new_prefix = prj.common_relocation(entry.npz_path, found)
        self.project.relocate_entry(entry, found)
        also = self.project.relocate_prefix(old_prefix, new_prefix)

        note = f"Repointed {entry.name}"
        note += f" and {len(also)} other(s) that moved with it. " if also else ". "
        self.refresh(note)


class QCWindow(QWidget):
    """Floating quality-control panel for one slide.

    Opened from the toolbar rather than docked: it is used once per slide, so
    it does not earn permanent space in the right-hand column.
    """

    saved = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlag(Qt.Window)
        self.setWindowTitle("Quality Control")
        self.resize(340, 220)

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.slide_label = QLabel("")
        self.slide_label.setWordWrap(True)
        self.slide_label.setStyleSheet("font-weight: bold;")
        self.layout.addWidget(self.slide_label)

        # "None" is the cleared state, and is what removes the row from the sheet
        self.group = QButtonGroup(self)
        self.buttons = {}
        for label in ("None",) + prj.QC_STATUSES:
            button = QRadioButton(label)
            self.group.addButton(button)
            self.layout.addWidget(button)
            self.buttons[label] = button
        self.buttons["None"].setChecked(True)

        # a flag, not a status: tissue can be unusable and still be graded
        self.bad_tissue = QCheckBox("Bad tissue")
        self.layout.addWidget(self.bad_tissue)

        self.layout.addWidget(QLabel("Comments (optional)"))
        self.comments = QLineEdit()
        self.layout.addWidget(self.comments)

        # closing is the single commit point, so the button just closes
        self.save_button = QPushButton("Save && Close")
        self.save_button.pressed.connect(self.close)
        self.layout.addWidget(self.save_button)

    def status(self):
        """The selected status, or "" for None."""
        for label, button in self.buttons.items():
            if button.isChecked():
                return "" if label == "None" else label
        return ""

    def set_state(self, slide_name, record):
        self.slide_label.setText(slide_name or "")
        self.buttons.get(record["status"] or "None",
                         self.buttons["None"]).setChecked(True)
        self.bad_tissue.setChecked(record["bad_tissue"])
        self.comments.setText(record["comments"])

    def closeEvent(self, event):
        # the X and Save & Close are the same path, so neither loses the edit
        self.saved.emit()
        super().closeEvent(event)


class RangeSlider(QWidget):
    """One groove with two handles: the span between them is the kept window.

    Qt has no range slider, and two stacked sliders cost twice the height for
    a single conceptual control, so this draws its own.
    """

    valueChanged = Signal()

    HANDLE_R = 6
    GROOVE = 4

    def __init__(self, minimum=1, maximum=1000):
        super().__init__()
        self.minimum, self.maximum = minimum, maximum
        self._low, self._high = minimum, maximum
        self._drag = None
        self.setMinimumHeight(2 * self.HANDLE_R + 8)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def low(self):
        return self._low

    def high(self):
        return self._high

    def set_values(self, low, high):
        self._low, self._high = low, high
        self.update()

    # -- value <-> pixel ----------------------------------------------------

    def _x(self, value):
        """Pixel centre for a value, inset so the handles stay on screen."""
        span = max(self.width() - 2 * self.HANDLE_R, 1)
        frac = (value - self.minimum) / max(self.maximum - self.minimum, 1)
        return self.HANDLE_R + frac * span

    def _value(self, x):
        span = max(self.width() - 2 * self.HANDLE_R, 1)
        frac = (x - self.HANDLE_R) / span
        value = self.minimum + frac * (self.maximum - self.minimum)
        return int(round(min(max(value, self.minimum), self.maximum)))

    # -- painting -----------------------------------------------------------

    def paintEvent(self, event):
        qp = QPainter(self)
        qp.setRenderHint(QPainter.RenderHint.Antialiasing)
        y = self.height() / 2

        for colour, a, b in (("#cfcfcf", self.minimum, self.maximum),
                             (RANGE_SLIDER_COLOUR, self._low, self._high)):
            qp.setPen(QPen(QColor(colour), self.GROOVE, Qt.SolidLine, Qt.RoundCap))
            qp.drawLine(QPointF(self._x(a), y), QPointF(self._x(b), y))

        qp.setPen(QPen(QColor("#8c8c8c"), 1))
        qp.setBrush(QColor("white"))
        for value in (self._low, self._high):
            qp.drawEllipse(QPointF(self._x(value), y), self.HANDLE_R, self.HANDLE_R)

    # -- dragging -----------------------------------------------------------

    def mousePressEvent(self, event):
        x = event.position().x()
        # grab whichever handle is nearer the click
        self._drag = ("low" if abs(x - self._x(self._low))
                      <= abs(x - self._x(self._high)) else "high")
        self.mouseMoveEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag is None:
            return
        value = self._value(event.position().x())
        # the handles may meet but never cross
        if self._drag == "low":
            self._low = min(value, self._high)
        else:
            self._high = max(value, self._low)
        self.update()
        self.valueChanged.emit()

    def mouseReleaseEvent(self, event):
        self._drag = None


class LabeledSlider(QWidget):
    """A slider with a caption. The window sliders run 1..1000 against a
    feature's value range; the opacity sliders run 0..100 as a percentage."""

    def __init__(self, label, orientation, value, minimum=1, maximum=1000):
        super().__init__()
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.layout.setSpacing(1)
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.label = QLabel(label)
        self.layout.addWidget(self.label)

        self.slider = QSlider(orientation=orientation)
        self.slider.setMinimum(minimum)
        self.slider.setMaximum(maximum)
        self.slider.setSliderPosition(value)
        self.layout.addWidget(self.slider)


class HistogramCanvas(FigureCanvasQTAgg):
    """Histogram of the selected feature, with the kept window in blue."""

    def __init__(self):
        super().__init__(Figure(figsize=(3, 2)))
        self.ax = self.figure.add_subplot(111)
        self.figure.subplots_adjust(left=0.08, right=0.97, top=0.88, bottom=0.22)
        # without a floor the dock layout squashes the plot to a few pixels
        self.setMinimumHeight(160)
        self.values = self.label = None
        self.bars, self.centres = [], []

    def plot(self, values, lo, hi, label):
        """Draw the distribution, then colour the bars by the kept window.

        The bars are only rebuilt when the data changes: dragging the window
        recolours the existing ones, which avoids re-binning ~800k values on
        every mouse move.
        """
        if values is not self.values or label != self.label:
            self.ax.clear()
            self.bars, self.centres = [], []
            if values.size:
                _, edges, self.bars = self.ax.hist(
                    values, bins=HISTOGRAM_BINS,
                    edgecolor=HISTOGRAM_EDGE, linewidth=0.4)
                self.centres = (edges[:-1] + edges[1:]) / 2

            self.ax.set_title(label, fontsize=8)
            self.ax.tick_params(labelsize=7)
            self.ax.set_yticks([])
            self.values, self.label = values, label

        for bar, centre in zip(self.bars, self.centres):
            bar.set_facecolor(HISTOGRAM_IN_COLOUR if lo <= centre <= hi
                              else HISTOGRAM_OUT_COLOUR)
        self.draw_idle()


class OverlayWidget(QWidget):
    """Everything that tints the slide: the feature heatmap and the GM/WM
    masks, each with its own opacity."""

    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # the panel carries a lot of controls, so it is packed tighter than
        # Qt's defaults
        self.layout.setSpacing(4)
        self.layout.setContentsMargins(6, 4, 6, 4)

        # -- boundaries ----------------------------------------------------
        self.show_contours = QCheckBox("Show GM / WM Contours   (C)")
        self.show_contours.setChecked(True)
        self.layout.addWidget(self.show_contours)

        # -- GM/WM masks ---------------------------------------------------
        self.show_mask = QCheckBox("Show GM / WM Mask   (M)")
        self.layout.addWidget(self.show_mask)

        self.mask_key = QLabel(
            '<span style="color:#%02x%02x%02x">&#9632;</span> GM &nbsp;'
            '<span style="color:#%02x%02x%02x">&#9632;</span> WM'
            % (*MASK_COLOURS["gm"], *MASK_COLOURS["wm"]))
        self.layout.addWidget(self.mask_key)

        self.mask_alpha = LabeledSlider("Mask opacity", Qt.Horizontal,
                                        DEFAULT_MASK_OPACITY, minimum=0, maximum=100)
        self.layout.addWidget(self.mask_alpha)

        # -- feature heatmap ------------------------------------------------
        self.show_heatmap = QCheckBox("Show Feature Heatmap   (H)")
        self.layout.addWidget(self.show_heatmap)

        self.feature_combo = QComboBox()
        for name, _ in FEATURE_CHANNELS:
            self.feature_combo.addItem(name)
        self.layout.addWidget(self.feature_combo)

        self.histogram = HistogramCanvas()
        self.layout.addWidget(self.histogram)

        # one control for the kept window; it runs 1..1000 and is mapped onto
        # the channel's real value range, so it serves both features
        self.window = RangeSlider(1, 1000)
        self.layout.addWidget(self.window)

        self.range_label = QLabel("")
        self.layout.addWidget(self.range_label)

        self.heatmap_alpha = LabeledSlider("Heatmap opacity", Qt.Horizontal,
                                           DEFAULT_HEATMAP_OPACITY, minimum=0, maximum=100)
        self.layout.addWidget(self.heatmap_alpha)


class AnnotationWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.gm_layout = QHBoxLayout()
        self.layout.addLayout(self.gm_layout)

        self.annotate_gm_button = QPushButton("Annotate GM")
        self.gm_layout.addWidget(self.annotate_gm_button)

        self.annotate_adjacent_wm = QCheckBox("Annotate Adjacent WM?")
        self.gm_layout.addWidget(self.annotate_adjacent_wm)

        self.wm_layout = QHBoxLayout()
        self.layout.addLayout(self.wm_layout)

        self.gm_width_label = QLabel("GM Width")
        self.wm_layout.addWidget(self.gm_width_label)

        self.gm_width_spinbox = QSpinBox()
        self.gm_width_spinbox.setMinimum(200)
        self.gm_width_spinbox.setMaximum(2000)
        self.gm_width_spinbox.setValue(1000)
        self.gm_width_spinbox.setSingleStep(100)
        self.wm_layout.addWidget(self.gm_width_spinbox)

        self.annotate_wm_button = QPushButton("Annotate Deep WM")
        self.wm_layout.addWidget(self.annotate_wm_button)

        self.snap_to_seg = QCheckBox("Snap to Segmentations?")
        self.snap_to_seg.setChecked(True)
        self.layout.addWidget(self.snap_to_seg)

        # both checkboxes only matter to the cursor-snapping path, so they stay
        # greyed out while SNAP_TO_SEGMENTATION is off
        self.annotate_adjacent_wm.setEnabled(SNAP_TO_SEGMENTATION)
        self.snap_to_seg.setEnabled(SNAP_TO_SEGMENTATION)

        # the ROIs on this slide, so they can be picked out and removed the
        # same way slides are in the project panel
        self.roi_list = QListWidget()
        self.roi_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.roi_list.setMaximumHeight(110)
        self.layout.addWidget(self.roi_list)

        self.delete_roi_button = QPushButton("Delete Selected ROI")
        self.delete_roi_button.setEnabled(False)
        self.layout.addWidget(self.delete_roi_button)


class SegmentationWidget(QWidget):
    """Re-run the NEUSEG GM/WM segmentation and inspect the GMM behind it."""

    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.run_button = QPushButton("Run NEUSEG")
        self.layout.addWidget(self.run_button)

        # hidden until a run starts, so the panel stays quiet when idle
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.hide()
        self.layout.addWidget(self.progress)

        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.status.setStyleSheet("color: gray;")
        self.layout.addWidget(self.status)

        self.show_gmm_button = QPushButton("Show GMM Plot")
        self.show_gmm_button.setEnabled(False)
        self.layout.addWidget(self.show_gmm_button)


class DockWidget(QDockWidget):
    """A dock panel wrapping one of the control widgets above."""

    def __init__(self, name, widget_class):
        super().__init__(name)
        self.widget = widget_class()
        self.setWidget(self.widget)


# ---------------------------------------------------------------------------
# 5. NEUSEG re-run and the GMM viewer
# ---------------------------------------------------------------------------

class NeusegWorker(QThread):
    """Re-runs the NEUSEG GM/WM segmentation off the GUI thread.

    This mirrors the body of `neuseg.tissue.segment_wm` rather than calling it,
    for two reasons: the GM posterior the scatter plots is internal to that
    function, and running the stages here is what makes a real progress bar
    possible. Keep it in step if segment_wm changes.
    """

    progress = Signal(int, str)     # percent, stage description
    finished_ok = Signal(object)    # dict of results
    failed = Signal(str)

    # start percentages, weighted by how long each stage actually takes: the
    # GMM fit is ~87% of a run, the rest finishes in well under a second
    STAGES = ((2, "Preparing feature heatmap..."),
              (5, "Fitting GMM..."),
              (90, "Post-processing: CRF and island pruning..."),
              (97, "Tracing GM/WM contours..."))

    def __init__(self, features, tissue_mask, thumbnail, log_path):
        super().__init__()
        self.features = features
        self.tissue_mask = tissue_mask
        self.thumbnail = thumbnail
        self.log_path = log_path

    def run(self):
        try:
            # imported here so the annotator still starts if neuseg is absent
            sys.path.insert(0, os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))))
            from neuseg.tissue import run_gmm, post_process, render_contours

            self.progress.emit(*self.STAGES[0])
            logger = sana.logging.Logger('quiet', fpath=self.log_path)

            self.progress.emit(*self.STAGES[1])
            gm_prob, tissue_features = run_gmm(
                self.thumbnail, self.tissue_mask, self.features, logger=logger)

            # the island floor is given in mm2, so it needs the heatmap's own
            # microns per pixel: the slide's, scaled by both downsamples
            self.progress.emit(*self.STAGES[2])
            mpp = (float(logger.data['mpp'])
                   * logger.data['ds'][logger.data['thumbnail_level']]
                   * logger.data['ds_thumbnail'])
            gm_mask, wm_mask = post_process(gm_prob, tissue_features,
                                            mpp=mpp, logger=logger)

            self.progress.emit(*self.STAGES[3])
            tissue_bool = self.tissue_mask.img.squeeze().astype(bool)
            contours = render_contours(self.thumbnail, gm_mask, wm_mask, tissue_bool)

            self.progress.emit(100, "Done")
            self.finished_ok.emit({"gm_prob": gm_prob, "gm_mask": gm_mask,
                                   "wm_mask": wm_mask, "contours": contours,
                                   "tissue": tissue_bool})
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")


class GMMWindow(QWidget):
    """Floating window showing the GMM feature space the segmentation came from.

    One point per sampled tissue pixel: soma density against soma size, coloured
    by the GM posterior. Hovering a point tells the annotator which pixel it is,
    so the slide can show where that measurement came from.
    """

    hovered = Signal(object)        # (row, col) on the heatmap grid, or None
    closed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("NEUSEG - GMM feature space")
        self.setWindowFlag(Qt.Window)
        self.resize(560, 520)

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.canvas = FigureCanvasQTAgg(Figure(figsize=(5, 4)))
        self.ax = self.canvas.figure.add_subplot(111)
        self.canvas.figure.subplots_adjust(left=0.16, right=0.98, top=0.93, bottom=0.13)
        self.layout.addWidget(self.canvas)

        self.readout = QLabel("Hover a point to locate it on the slide")
        self.readout.setStyleSheet("color: gray;")
        self.layout.addWidget(self.readout)

        self.coords = None          # (row, col) per plotted point
        self.canvas.mpl_connect("motion_notify_event", self.on_motion)

    def set_data(self, density, size, gm_prob, coords, title):
        """Draw the sampled cloud. Arrays are parallel and already subsampled."""
        self.coords = coords
        self.ax.clear()
        dots = self.ax.scatter(density, size, c=gm_prob, cmap=GMM_COLORMAP,
                               s=4, alpha=0.5, linewidths=0, vmin=0, vmax=1)
        if not getattr(self, "colorbar", None):
            self.colorbar = self.canvas.figure.colorbar(dots, ax=self.ax)
            self.colorbar.set_label("WM  <-  posterior  ->  GM", fontsize=8)
        self.ax.set_xlabel("soma density", fontsize=9)
        self.ax.set_ylabel("soma size", fontsize=9)
        self.ax.set_title(title, fontsize=9)
        self.ax.tick_params(labelsize=8)

        # two markers, drawn once and moved thereafter: one follows the mouse
        # inside this plot, the other follows the mouse over the slide
        self.cursor, = self.ax.plot([], [], "o", mfc="none", mec="black", mew=2, ms=10)
        self.slide_cursor, = self.ax.plot([], [], "o", mfc="none", mec="#0288d1",
                                          mew=2, ms=13)
        self.canvas.draw_idle()

    def mark_feature(self, density, size):
        """Ring the point for a pixel hovered on the slide.

        Plotted from the pixel's own feature values rather than searched for in
        the sample, so it is exact even though only a sample is drawn.
        """
        if self.coords is None:
            return
        self.slide_cursor.set_data([density], [size])
        self.canvas.draw_idle()

    def closeEvent(self, event):
        # lets the annotator drop the mouse tracking it turned on for us
        self.closed.emit()
        super().closeEvent(event)

    def on_motion(self, event):
        """Report the nearest point, in axis units so both features count equally."""
        if self.coords is None or event.inaxes is not self.ax:
            return
        offsets = self.ax.collections[0].get_offsets()

        # normalise by the axis span, otherwise the larger-valued feature
        # decides which point is "nearest"
        (x0, x1), (y0, y1) = self.ax.get_xlim(), self.ax.get_ylim()
        d = (((offsets[:, 0] - event.xdata) / (x1 - x0)) ** 2
             + ((offsets[:, 1] - event.ydata) / (y1 - y0)) ** 2)
        i = int(np.argmin(d))

        row, col = self.coords[i]
        self.cursor.set_data([offsets[i, 0]], [offsets[i, 1]])
        self.readout.setText(f"density {offsets[i, 0]:.4g}   size {offsets[i, 1]:.4g}"
                             f"   at pixel (x={col}, y={row})")
        self.canvas.draw_idle()
        self.hovered.emit((row, col))


# ---------------------------------------------------------------------------
# 6. GeoJSON I/O
# ---------------------------------------------------------------------------

# removes unreadable header data from JSON annotation files
# NOTE: these headers come export JSON files from Qupath
def fix_annotations(ifile):

    # load the data as bytes
    with open(ifile, 'rb') as fp:
        data = fp.read()

    # find the index of the first annotation in the json
    ind = data.find(b'[\n')
    if ind == -1:
        ind = data.find(b'[]')
        if ind == -1:
            return

    # rewrite the data starting at the first annotation
    with open(ifile, 'wb') as fp:
        fp.write(data[ind:])
#
# end of fix_annotation


# pulls the xy coordinates out of one GeoJSON geometry
# NOTE: MultiPolygon rings are concatenated into a single coordinate list
# TODO: this should be simplified.
#        need to actually handle what a MultiPolygon is
def geometry_to_xy(geo):

    if geo['type'] == 'MultiPolygon':
        x, y = [], []
        for coords in geo['coordinates']:
            x += [float(c[0]) for c in coords[0]]
            y += [float(c[1]) for c in coords[0]]
        return np.array(x), np.array(y)

    if geo['type'] == 'Polygon':
        coords = geo['coordinates'][0]
    elif geo['type'] in ('MultiPoint', 'LineString'):
        coords = geo['coordinates']
    else:
        return np.array([]), np.array([])

    return (np.array([float(c[0]) for c in coords]),
            np.array([float(c[1]) for c in coords]))
#
# end of geometry_to_xy


# loads a JSON annotation file into memory
#  -ifile: input JSON file to be read
#  -class_name: if given, only returns annotations with this class
#  -annotation_name: if given, only returns annotations with this name
def read_annotations(ifile, class_name=None, annotation_name=None):

    if ifile.endswith('.geojson'):
        data = geojson.load(open(ifile, 'r'))
        if hasattr(data, 'features'):
            data = data['features']

    elif ifile.endswith('.json'):

        # blank data if the file doesn't exist
        if not os.path.exists(ifile):
            return []

        # remove unwanted header bytes if they exist
        fix_annotations(ifile)

        # load the json data
        with open(ifile, 'r', encoding='utf-8') as fp:
            data = json.loads(fp.read())

    else:
        raise Exception

    # load the annotations
    # NOTE: this could be handled by a GeoJSON package?
    annotations = []
    for annotation in data:
        properties = annotation['properties']

        # class name, annotation name and attributes are all optional
        classification = properties.get('classification')
        cname = classification['name'] if classification else ""
        aname = properties.get('name', "")
        attributes = properties.get('attributes', {})

        x, y = geometry_to_xy(annotation['geometry'])
        annotations.append(
            sana.geo.Annotation(x, y, ifile, cname, aname,
                                attributes=attributes, is_micron=False, level=0))
    #
    # end of annotation loop

    # only return annotations matching the given class/annotation name
    if class_name is not None:
        annotations = [a for a in annotations
                       if fnmatch.fnmatch(a.class_name, class_name)]
    if annotation_name is not None:
        annotations = [a for a in annotations
                       if fnmatch.fnmatch(a.annotation_name, annotation_name)]

    return annotations
#
# end of read_annotations


# writes a list of Polygon annotations to a JSON annotation file
#  -ofile: location to write the annotations to
#  -annos: list of Polygon Annotations
def write_annotations(ofile, annos):

    # convert the Ann objects to json strings
    json_annos = [anno.to_geojson() for anno in annos]

    # write the file
    with open(ofile, 'w') as fp:
        json.dump(json_annos, fp, indent=2)
#
# end of write_annotations


# ---------------------------------------------------------------------------
# 7. main
# ---------------------------------------------------------------------------

def main(argv):
    project_path = argv[1] if len(argv) > 1 else None
    start_idx = int(argv[2]) if len(argv) > 2 else 0

    app = QApplication(argv)

    # a project given on the command line skips the startup prompt; anything
    # else (e.g. a bare slide folder) falls through to it rather than crashing
    project = None
    if project_path:
        try:
            project = prj.Project.open(project_path)
        except Exception as e:
            print(f"Not a project folder, opening the project chooser: {e}")

    # the window must stay referenced: a top-level widget is owned by Python,
    # so dropping the last reference destroys it and app.exec() returns at once
    window = SlideAnnotator(project=project, start_idx=start_idx, do_ask_name=False)
    window.show()

    sys.exit(app.exec())


if __name__ == '__main__':
    main(sys.argv)
