#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import fnmatch
import geojson
from tqdm import tqdm

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt, QPoint, QLine, QRectF
from PySide6.QtGui import QImage, QPixmap, QPalette, QPainter, QAction, QPen, QFont
from PySide6.QtPrintSupport import QPrintDialog, QPrinter
from PySide6.QtWidgets import QLabel, QSizePolicy, QScrollArea, QMessageBox, QMainWindow, QMenu, QFileDialog, QStyle, QVBoxLayout,QHBoxLayout, QComboBox, QCheckBox, QWidget, QDockWidget, QSlider, QToolBar, QPushButton, QLineEdit, QSpinBox

from matplotlib import pyplot as plt

import PIL.Image
import PIL.ImageDraw
import numpy as np
import cv2
import pdnl_sana as sana
import pdnl_sana.image
import pdnl_sana.interpolate
import pdnl_sana.process

class Annotation:
    def __init__(self, is_gm, is_wm):
        self.csf0 = None
        self.csf1 = None
        self.wm0 = None
        self.wm1 = None

        self.csf_poly = None
        self.csf0_idx = None
        self.csf1_idx = None

        self.wm_poly = None
        self.wm0_idx = None
        self.wm1_idx = None

        self.wm2 = None
        self.wm3 = None

        self.is_gm = is_gm
        self.is_wm = is_wm
        self.saved = False
        self.done = False
        self.name = ""

    def save(self):
        self.saved = True
        if self.is_gm:
            if not self.csf_poly is None:
                csf = self.csf_poly.slice_shortest(self.csf0_idx, self.csf1_idx).astype(float)
                csf = sana.interpolate.fit_rotated_polynomial(csf, 3, 20)
                if csf is None:
                    csf = self.csf_poly[np.array([self.csf0_idx, self.csf1_idx])].to_curve()
            else:
                csf = sana.geo.Curve([self.csf0.x(), self.csf1.x()], [self.csf0.y(), self.csf1.y()]).astype(float)

            if not self.wm_poly is None:
                gm = self.wm_poly.slice_shortest(self.wm0_idx, self.wm1_idx).astype(float)
                gm = sana.interpolate.fit_rotated_polynomial(gm, 3, 20)
                if gm is None:
                    gm = self.wm_poly[np.array([self.wm0_idx, self.wm1_idx])].to_curve()
            else:
                gm = sana.geo.Curve([self.wm0.x(), self.wm1.x()], [self.wm0.y(), self.wm1.y()]).astype(float)

            s0 = sana.geo.curve_like(csf, [self.csf0.x(), self.wm0.x()], [self.csf0.y(), self.wm0.y()]).astype(float)
            s1 = sana.geo.curve_like(csf, [self.csf1.x(), self.wm1.x()], [self.csf1.y(), self.wm1.y()]).astype(float)

            ctr = sana.geo.point_like(csf, 0, 0)
            angle = csf.get_angle()
            [x.rotate(ctr, -angle) for x in [csf, gm, s0, s1]]
            if np.mean(csf[:,1]) > np.mean(gm[:,1]):
                angle += 180
                [x.rotate(ctr, 180) for x in [csf, gm, s0, s1]]
            if np.mean(s0[:,0]) < np.mean(s1[:,0]):
                l = s0
                r = s1
            else:
                l = s1
                r = s0
            if csf[0,0] > csf[-1,0]: csf = csf[::-1]
            if r[0,1] > r[-1,1]: r = r[::-1]
            if gm[0,0] < gm[-1,0]: gm = gm[::-1]
            if l[0,1] < l[-1,1]: l = l[::-1]
            [x.rotate(ctr, angle) for x in [csf, gm, s0, s1]]

            self.csf_gm_seg = csf
            self.gm_wm_seg = gm
            self.left_wall = l
            self.right_wall = r
        
            if self.is_wm:
                test_wm_roi_0 = sana.geo.Polygon(
                    [*self.gm_wm_seg[:,0], self.wm2.x(), self.wm3.x()], 
                    [*self.gm_wm_seg[:,1], self.wm2.y(), self.wm3.y()], 
                ).astype(float)
                test_wm_roi_1 = sana.geo.Polygon(
                    [*self.gm_wm_seg[:,0], self.wm3.x(), self.wm2.x()], 
                    [*self.gm_wm_seg[:,1], self.wm3.y(), self.wm2.y()], 
                ).astype(float)
                if test_wm_roi_0.get_area() > test_wm_roi_1.get_area():
                    self.wm_roi = test_wm_roi_0.connect()
                else:
                    self.wm_roi = test_wm_roi_1.connect()
        else:
            self.wm_roi = sana.geo.Polygon(
                [self.wm0.x(), self.wm1.x(), self.wm2.x(), self.wm3.x()], 
                [self.wm0.y(), self.wm1.y(), self.wm2.y(), self.wm3.y()], 
            ).astype(float).connect()

class Canvas(QLabel):
    def __init__(self, mask):
        super().__init__()
        self.point = None
        self.idx = None
        self.poly = None
        self.annotations = []
        self.current_annotation = None
        self.radius = 4
        self.scale_factor = 1.0
        self.mask = mask

    def reset_annotations(self):
        self.annotations = []
        self.current_annotation = None
        self.update()

    def get_shortest(self, gm, ctr, n_angles=360, debug=False):

        dists, pts = [], []
        for theta in np.linspace(0, np.pi, n_angles):
            csf, wm = None,None
        
            r = 0
            direction = +1
            while True:
                r += direction
                x = int(r*np.cos(theta) + ctr[0])
                y = int(r*np.sin(theta) + ctr[1])
                if x < 0 or y < 0 or x >= gm.img.shape[1] or y >= gm.img.shape[0]:
                    break            
                if csf is None and gm.img[y,x] == 0:
                    csf = (x,y)
                    break
                if wm is None and gm.img[y,x] == 2:
                    wm = (x,y)
                    break

            r = 0
            direction = -1
            while True:
                r += direction
                x = int(r*np.cos(theta) + ctr[0])
                y = int(r*np.sin(theta) + ctr[1])
                if x < 0 or y < 0 or x >= gm.img.shape[1] or y >= gm.img.shape[0]:
                    break
                if csf is None and gm.img[y,x] == 0:
                    csf = (x,y)
                    break
                if wm is None and gm.img[y,x] == 2:
                    wm = (x,y)
                    break

            if csf is None or wm is None:
                continue
        
            dists.append(np.sqrt((csf[0]-wm[0])**2+(csf[1]-wm[1])**2))
            pts.append([csf,wm])

        return pts[np.argmin(dists)]
        
    def get_roi(self, gm, ctr, l):
        x, y = ctr

        # get the spine of the ROI
        csf, wm = self.get_shortest(gm, (x,y))
    
        # get the lateral direction of cortex
        th = np.arctan2((wm[1]-csf[1]), (wm[0]-csf[0]))
        thp = th + np.pi/2
    
        p0 = (-l*np.cos(thp)+ctr[0],-l*np.sin(thp)+ctr[1])
        p1 = (l*np.cos(thp)+ctr[0],l*np.sin(thp)+ctr[1])

        # get the left and right boundaries
        csf0, wm0 = self.get_shortest(gm, p0)
        csf1, wm1 = self.get_shortest(gm, p1)

        # get the polygon vertices to sample at
        # gm_mask = gm.copy()
        # gm_mask.img[gm_mask.img == 2] = 0
        # gm_polys = gm_mask.to_polygons()[0]
        gm_polys = self.gm_polys
        sample_pts = []
        sample_idxs = []

        dists, csf_idxs = [], []
        for poly in gm_polys:
            d0 = np.sqrt((poly[:,0]-csf0[0])**2+(poly[:,1]-csf0[1])**2)
            d1 = np.sqrt((poly[:,0]-csf1[0])**2+(poly[:,1]-csf1[1])**2)
            csf_idxs.append([np.argmin(d0),np.argmin(d1)])
            dists.append(np.min(d0)+np.min(d1))
        csf_idx_0, csf_idx_1 = csf_idxs[np.argmin(dists)]
        csf_sample_poly = gm_polys[np.argmin(dists)]

        dists, wm_idxs = [], []
        for poly in gm_polys:
            d0 = np.sqrt((poly[:,0]-wm0[0])**2+(poly[:,1]-wm0[1])**2)
            d1 = np.sqrt((poly[:,0]-wm1[0])**2+(poly[:,1]-wm1[1])**2)
            wm_idxs.append([np.argmin(d0),np.argmin(d1)])
            dists.append(np.min(d0)+np.min(d1))
        wm_idx_0, wm_idx_1 = wm_idxs[np.argmin(dists)]
        wm_sample_poly = gm_polys[np.argmin(dists)]

        def slice_shortest(p, i, j, level):
            if j < i:
                (i,j) = (j,i)
            opt1 = p[i:j]
            opt2 = np.concatenate([p[j:],p[:i]])
            opt1 = pdnl_sana.geo.Curve(*opt1.T, is_micron=False, level=level)
            opt2 = pdnl_sana.geo.Curve(*opt2.T, is_micron=False, level=level)        
            if opt1.get_length() < opt2.get_length():
                return opt1
            else:
                return opt2
        #csf_gm = slice_shortest(csf_sample_poly, csf_idx_0, csf_idx_1, level=gm.level)
        #gm_wm = slice_shortest(wm_sample_poly, wm_idx_0, wm_idx_1, level=gm.level)

        #return csf_gm, gm_wm, [[csf,wm],[p0,p1],[csf0,wm0],[csf1,wm1]]
        return csf_sample_poly, csf_idx_0, csf_idx_1, wm_sample_poly, wm_idx_0, wm_idx_1
        
    def select_point(self):
        a = self.current_annotation
        if a is None:
            return

        if a.is_gm:
            x = self.point.x()
            y = self.point.y()
            csf_poly, csf0, csf1, wm_poly, wm0, wm1 = self.get_roi(self.mask, (x,y), l=self.gm_width//(2*16*.5045))
            a.csf_poly = csf_poly
            a.csf0_idx = csf0
            a.csf0 = QPoint(*csf_poly[csf0])
            a.wm_poly = wm_poly
            a.wm0_idx = wm0
            a.wm0 = QPoint(*wm_poly[wm0])
            a.csf1_idx = csf1
            a.csf1 = QPoint(*csf_poly[csf1])
            a.wm1_idx = wm1
            a.wm1 = QPoint(*wm_poly[wm1])
            a.done = True
            # if a.csf0 is None:
            #     a.csf0 = self.point
            #     a.csf0_idx = self.idx
            #     a.csf_poly = self.poly.copy() if not self.poly is None else None
            # elif a.wm0 is None:
            #     a.wm0 = self.point
            #     a.wm0_idx = self.idx
            #     a.wm_poly = self.poly.copy() if not self.poly is None else None
            # elif a.csf1 is None:
            #     a.csf1 = self.point
            #     a.csf1_idx = self.idx
            # elif a.wm1 is None:
            #     a.wm1 = self.point
            #     a.wm1_idx = self.idx
            #     if not a.is_wm:
            #         a.done = True
            # elif a.is_wm and a.wm2 is None:
            #     a.wm2 = self.point
            # elif a.is_wm and a.wm3 is None:
            #     a.wm3 = self.point
            #     a.done = True

        elif a.is_wm:
            if a.wm0 is None:
                a.wm0 = self.point
            elif a.wm1 is None:
                a.wm1 = self.point
            elif a.wm2 is None:
                a.wm2 = self.point
            elif a.wm3 is None:
                a.wm3 = self.point
                a.done = True
        self.update()

    def draw_circle(self, p0):
        if p0 is None:
            return
        self.qp.drawEllipse(p0 * self.scale_factor, self.radius, self.radius)
    def draw_line(self, p0, p1):
        if p0 is None or p1 is None:
            return
        self.qp.drawLine(QLine(p0*self.scale_factor, p1*self.scale_factor))

    # TODO: arc should be stored somewhere and we just plot it
    def draw_arc(self, poly, i0, i1):
        if poly is None or i0 is None or i1 is None:
            return
        if i0 == i1:
            return
        c = poly.slice_shortest(i0, i1)
        smooth_c = sana.interpolate.fit_rotated_polynomial(c, 3, 10)
        if not smooth_c is None:
            self.draw_curve(smooth_c)
        else:
            self.draw_curve(poly.slice_shortest(i0, i1))

    def draw_curve(self, c):
        for i in range(len(c)-1):
            a = QPoint(*(c[i]))
            b = QPoint(*(c[i+1]))
            self.draw_line(a, b)

    def draw_text(self, x, y, text):
        h = self.radius
        w = len(text)*self.radius

        # self.qp.drawText(QRectF((x-w//2)*self.scale_factor, (y-h//2)*self.scale_factor, w, h), Qt.AlignmentFlag.AlignCenter, text)
        self.qp.drawText(QPoint((x-w//2)*self.scale_factor, (y-h//2)*self.scale_factor), text)

    # TODO: this should be drawn on the image itself once saved
    # TODO: then reset the image if reseting annotations
    def draw_annotation(self, a):
        if a.saved:
            self.qp.setPen(QPen(Qt.red, self.radius))
            if a.is_gm:
                roi = sana.geo.polygon_like(a.csf_gm_seg, *np.concatenate([a.csf_gm_seg, a.right_wall, a.gm_wm_seg, a.left_wall], axis=0).T)
                cx, cy = roi.get_centroid()
                self.draw_curve(roi)
                self.draw_text(cx, cy, a.name)
            if a.is_wm:
                self.draw_curve(a.wm_roi)
                cx, cy = a.wm_roi.get_centroid()
                self.draw_text(cx, cy, a.name)
            
            self.qp.setPen(QPen(Qt.green, self.radius))            
        else:
            [self.draw_circle(p) for p in [a.csf0, a.csf1, a.wm0, a.wm1, a.wm2, a.wm3]]
            self.draw_line(a.csf0, a.wm0)
            self.draw_line(a.csf1, a.wm1)
            if not a.csf_poly is None:
                self.draw_arc(a.csf_poly, a.csf0_idx, a.csf1_idx)
            else:
                self.draw_line(a.csf0, a.csf1)
            if a.is_gm:
                if not a.wm_poly is None:
                    self.draw_arc(a.wm_poly, a.wm0_idx, a.wm1_idx)
                else:
                    self.draw_line(a.wm0, a.wm1)
            else:
                self.draw_line(a.wm0, a.wm1)
            self.draw_line(a.wm1, a.wm2)
            self.draw_line(a.wm2, a.wm3)
            self.draw_line(a.wm3, a.wm0)

    def paintEvent(self, event):
        super().paintEvent(event)

        self.qp = QPainter(self)
        self.qp.setPen(QPen(Qt.green, self.radius))
        self.qp.setBrush(Qt.black)

        for annotation in self.annotations:
            self.draw_annotation(annotation)

        if not self.point is None:
            self.draw_circle(self.point)

        a = self.current_annotation
        if a is None:
            self.qp = None
            return
        self.draw_annotation(a)

        if a.is_gm:
            # live first wall
            if not a.csf0 is None and a.wm0 is None:
                self.draw_line(a.csf0, self.point)

            # live csf boundary
            if not a.csf0 is None and not a.wm0 is None and a.csf1 is None:
                if not a.csf_poly is None:
                    self.draw_arc(a.csf_poly, a.csf0_idx, self.idx)
                else:
                    self.draw_line(a.csf0, self.point)

            # live second wall
            if not a.csf1 is None and a.wm1 is None:
                self.draw_line(a.csf1, self.point)

            # live wm boundary
            if not a.wm0 is None and not a.csf1 is None and a.wm1 is None:
                if not a.wm_poly is None:
                    self.draw_arc(a.wm_poly, a.wm0_idx, self.idx)
                else:
                    self.draw_line(a.wm0, self.point)

            # live first wm wall
            if a.is_wm and not a.wm1 is None and a.wm2 is None:
                self.draw_line(a.wm1, self.point)

            # live second/third wm wall
            if a.is_wm and not a.wm2 is None and a.wm3 is None:
                self.draw_line(a.wm2, self.point)
                self.draw_line(a.wm0, self.point)

        elif a.is_wm:
            if not a.wm0 is None and a.wm1 is None:
                self.draw_line(a.wm0, self.point)
            if not a.wm1 is None and a.wm2 is None:
                self.draw_line(a.wm1, self.point)
            if not a.wm2 is None and a.wm3 is None:
                self.draw_line(a.wm2, self.point)
                self.draw_line(a.wm0, self.point)

        self.qp = None



class SlideAnnotator(QMainWindow):

    FILE_NAMES = {
        "Thumbnail": "thumbnail.png",\
        "Features": "feature_heatmap_tb_reso.npy",
        "Train Features": "feature_heatmap.npy",
        "GM Mask": "gm_mask.npy",
        "WM Mask": "wm_mask.npy",
        "Tissue Mask": "tissue_mask.npy",
        "Annotation": "annotations.geojson",
    }
    IGNORE_CHECK = ["Annotation"]

    def __init__(self, basedir=None, start_idx=0, do_ask_name=True):
        super().__init__()

        self.SCROLLBAR_EXTENT = QApplication.style().pixelMetric(QStyle.PixelMetric.PM_ScrollBarExtent)
        self.TITLEBAR_HEIGHT = QApplication.style().pixelMetric(QStyle.PixelMetric.PM_TitleBarHeight)

        self.DESKTOP_RECT = QApplication.primaryScreen().availableGeometry()
        self.DESKTOP_HEIGHT = self.DESKTOP_RECT.height()
        self.DESKTOP_WIDTH = self.DESKTOP_RECT.width()

        self.scale_factor = 1.0

        self.segmentation_mask = None
        self.canvas = Canvas(self.segmentation_mask)
        self.canvas.setBackgroundRole(QPalette.ColorRole.Base)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.canvas.setScaledContents(True)

        self.scroll_area = QScrollArea()
        # self.scroll_area.setBackgroundRole(QPalette.Dark)
        self.scroll_area.setWidget(self.canvas)
        self.scroll_area.setVisible(False)
        self.scroll_area.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignHCenter)

        self.setCentralWidget(self.scroll_area)

        self.setWindowTitle("SANA ROI Annotator")
        self.resize(800, 600)

        self.init_overlay_dock()
        self.init_annotation_dock()
        self.init_segmentation_dock()
        self.init_navigation()

        self.create_actions()
        self.create_menus()

        self.is_annotating = False
        self.is_segmenting_gm = False
        self.is_segmenting_wm = False
        self.init_gm = None
        self.init_wm = None
        
        self.do_ask_name = do_ask_name
        
        if basedir is None:
            basedir = self.open_directory()
        self.init_basedir(basedir, start_idx)


        
        self.show()
        self.setFocus()

    def toggle_mouse_tracking(self, state):
        self.setMouseTracking(state)
        self.canvas.setMouseTracking(state)
        self.scroll_area.setMouseTracking(state)
        self.canvas.setFocus()
        self.canvas.point = None
        self.canvas.update()

    def toggle_is_annotating(self, state=None):
        if state:
            self.is_annotating = state
        else:
            self.is_annotating = not self.is_annotating
        self.toggle_mouse_tracking(self.is_annotating)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.toggle_is_annotating(False)
            self.canvas.current_annotation = None
        if event.key() == Qt.Key_A:
            self.start_annotating_gm()
        if event.key() == Qt.Key_W:
            self.start_annotating_wm()
        if event.key() == Qt.Key_N or event.key() == Qt.Key_Space:
            self.load_next_slide()
        if event.key() == Qt.Key_P:
            self.load_previous_slide()

    def save_current_annotation(self):
        
        self.current_saving_annotation = self.canvas.current_annotation
        self.toggle_is_annotating(False)
                
        if self.do_ask_name:
            self.ask_name = QWidget()
            self.ask_name.vlayout = QVBoxLayout()
            self.ask_name.setLayout(self.ask_name.vlayout)

            self.ask_name.name = QLineEdit("")
            self.ask_name.vlayout.addWidget(self.ask_name.name)

            self.ask_name.save = QPushButton("Save")
            self.ask_name.vlayout.addWidget(self.ask_name.save)
            self.ask_name.save.pressed.connect(self.confirm_save_current_annotation)

            self.ask_name.cancel = QPushButton("Cancel")
            self.ask_name.vlayout.addWidget(self.ask_name.cancel)
            self.ask_name.cancel.pressed.connect(self.cancel_save_current_annotation)

            self.ask_name.show()
        else:

            if self.current_saving_annotation.is_gm:
                roi_name = f"GM_{len(self.canvas.annotations)}"
            else:
                roi_name = f"WM_{len(self.canvas.annotations)}"
            self.confirm_save_current_annotation(roi_name)

    def confirm_save_current_annotation(self, roi_name=None):
        if roi_name is None:
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

    def cancel_save_current_annotation(self):
        self.current_saving_annotation = None
        self.ask_name.hide()
        self.ask_name = None
        self.canvas.current_annotation = None

    def save_annotations(self):
        annotations = []
        for a in self.canvas.annotations:
            if a.is_gm:
                csf = a.csf_gm_seg.to_annotation(class_name="CSF", annotation_name=a.name)
                gm = a.gm_wm_seg.to_annotation(class_name="GM", annotation_name=a.name)
                l = a.left_wall.to_annotation(class_name="L", annotation_name=a.name)
                r = a.right_wall.to_annotation(class_name="R", annotation_name=a.name)
                annotations += [csf, r, gm, l]
            if a.is_wm:
                wm = a.wm_roi.to_annotation(class_name="ROI", annotation_name=a.name)
                annotations.append(wm)
        ofile = os.path.join(self.basedir, self.slide_dir, self.FILE_NAMES["Annotation"])
        write_annotations(ofile, annotations)
            
    def load_annotations(self, annotations):
        for name in set([x.annotation_name for x in annotations]):
            annotation = Annotation(is_gm=False, is_wm=False)
            annotation.name = name
            annotation.saved = True

            annos = [x for x in annotations if x.annotation_name == name]
            try:
                annotation.is_gm = True
                annotation.csf_gm_seg = [x for x in annos if x.class_name == 'CSF'][0].to_curve()
                annotation.right_wall  = [x for x in annos if x.class_name == 'R'][0].to_curve()
                annotation.gm_wm_seg  = [x for x in annos if x.class_name == 'GM'][0].to_curve()
                annotation.left_wall  = [x for x in annos if x.class_name == 'L'][0].to_curve()
            except:
                annotation.is_gm = False
            try:
                annotation.is_wm = True
                annotation.wm_roi = [x for x in annos if x.class_name == 'ROI'][0].to_polygon()
            except:
                annotation.is_wm = False
            self.canvas.annotations.append(annotation)
        self.canvas.update()

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)

        p = event.position()
        p = np.array([
            p.x() - self.canvas.pos().x(), 
            p.y() - self.canvas.pos().y() - self.navigation_toolbar.height() - self.TITLEBAR_HEIGHT,
        ]) / self.scale_factor

        if self.is_segmenting_gm:
            self.canvas.point = QPoint(p[0], p[1])
            self.canvas.update()
            return
        if self.is_segmenting_wm:
            self.canvas.point = QPoint(p[0], p[1])
            self.canvas.update()
            return
        else:
            self.canvas.point = QPoint(p[0], p[1])
            self.canvas.update()
            return
        
        a = self.canvas.current_annotation
        if a is None:
            return

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
                if a.wm2 is None:
                    for poly in self.wm_polys:
                        if sana.geo.ray_tracing(p[0], p[1], poly):
                            self.canvas.point = QPoint(p[0], p[1])
                            self.canvas.update()
                            return
                    self.canvas.point = None
                    self.canvas.update()
                    return
                elif a.wm3 is None:
                    for poly in self.wm_polys:
                        if sana.geo.ray_tracing(p[0], p[1], poly):
                            self.canvas.point = QPoint(p[0], p[1])
                            self.canvas.update()
                            return
                    self.canvas.point = None
                    self.canvas.update()
                    return
                else:
                    return
            else:
                return
        elif self.annotation_dock.widget.snap_to_seg.isChecked():
            if a.wm3 is None:
                for poly in self.wm_polys:
                    if sana.geo.ray_tracing(p[0], p[1], poly):
                        self.canvas.point = QPoint(p[0], p[1])
                        self.canvas.update()
                        return
                self.canvas.point = None
                self.canvas.update()
                return
            else:
                return
        else:
            pass
            
            
        if self.annotation_dock.widget.snap_to_seg.isChecked() or ((a.wm0 is None and not a.csf0 is None) or (not a.csf1 is None and a.wm1 is None)):
            idxs = []
            dists = []
            for poly in available_polys:
                d = np.sum((p - poly)**2, axis=1)
                idxs.append(np.argmin(d))
                dists.append(np.min(d))
            poly_idx = np.argmin(dists)
            poly = available_polys[poly_idx]
            idx = idxs[poly_idx]
            v = poly[idx]
            v = QPoint(v[0], v[1])
            self.canvas.point = v
            self.canvas.idx = idx
            self.canvas.poly = poly
            self.canvas.update()
        else:
            self.canvas.point = QPoint(p[0], p[1])
            self.canvas.idx = None
            self.canvas.poly = None
            self.canvas.update()

    # TODO: add click/drag logic for 4 control points
    def mousePressEvent(self, event):
        p = event.position()
        if event.button() == Qt.LeftButton and p.x() < self.canvas.width() and p.y() < self.canvas.height():
            if self.is_segmenting_gm:
                self.init_gm = self.canvas.point
                self.is_segmenting_gm = False
                self.toggle_mouse_tracking(False)
            if self.is_segmenting_wm:
                self.init_wm = self.canvas.point
                self.is_segmenting_wm = False
                self.toggle_mouse_tracking(False)
            if self.is_annotating:
            #    print('asdfjklasdfjklsjkla')
               self.canvas.select_point()
               if self.canvas.current_annotation is None or self.canvas.current_annotation.done:
                   self.save_current_annotation()

        # if event.button() == Qt.RightButton:
        #     self.canvas.annotations = []
        #     self.canvas.current_annotation = None
        #     self.canvas.update()

    # recursively finds directories under basedir that directly contain
    # all the required preprocessed slide files, at any nesting depth
    # (e.g. basedir/slide/... or basedir/category/patient/slide/slide/...)
    def find_slide_dirs(self, basedir):
        required = [self.FILE_NAMES[name] for name in self.FILE_NAMES if name not in self.IGNORE_CHECK]

        slide_dirs = []
        for dirpath, dirnames, filenames in os.walk(basedir):
            dirnames.sort()
            if dirpath == basedir:
                continue
            filenames = set(filenames)
            if all(f in filenames for f in required):
                slide_dirs.append(os.path.relpath(dirpath, basedir))
                # a matched slide dir shouldn't contain further nested slides
                dirnames[:] = []

        return sorted(slide_dirs)

    def init_basedir(self, basedir, start_idx):
        if basedir is None:
            dlg = QMessageBox(self)
            dlg.setWindowTitle("Error")
            dlg.setText("Directory not selected")
            button = dlg.exec()
            return

        slide_names = []
        slide_dirs = []
        warnings = []
        for slide_dir in tqdm(self.find_slide_dirs(basedir), desc="initalizing preprocessed data"):
            slide_name = os.path.basename(slide_dir)
            if os.path.exists(os.path.join(basedir, f'[REVIEW]_{slide_name}.png')):
                continue
            if os.path.exists(os.path.join(basedir, f'[WARNING]_{slide_name}.png')):
                warn = True
            else:
                warn = False

            slide_names.append(slide_name)
            slide_dirs.append(slide_dir)
            warnings.append(warn)

        if len(slide_names) == 0:
            dlg = QMessageBox(self)
            dlg.setWindowTitle("Error")
            dlg.setText("Directory does not contain preprocessed slides!")
            button = dlg.exec()
            return

        self.basedir = basedir
        self.slide_names = slide_names
        self.slide_dirs = slide_dirs
        self.warnings = warnings

        self.load_slide(start_idx)

    def init_overlay_dock(self):
        self.overlay_dock = OverlayDockWidget("Feature Overlays")
        self.overlay_dock.hide()
        self.overlay_dock.setAllowedAreas(Qt.DockWidgetArea.RightDockWidgetArea)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.overlay_dock)

        self.overlay_dock.widget.density_overlay.min.slider.valueChanged.connect(self.overlay_features)
        self.overlay_dock.widget.density_overlay.max.slider.valueChanged.connect(self.overlay_features)
        self.overlay_dock.widget.size_overlay.min.slider.valueChanged.connect(self.overlay_features)
        self.overlay_dock.widget.size_overlay.max.slider.valueChanged.connect(self.overlay_features)

    def init_annotation_dock(self):
        self.annotation_dock = AnnotationDockWidget("Annotation Inteface")
        self.annotation_dock.hide()
        self.annotation_dock.setAllowedAreas(Qt.DockWidgetArea.RightDockWidgetArea)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.annotation_dock)

        self.annotation_dock.widget.annotate_gm_button.pressed.connect(self.start_annotating_gm)
        self.annotation_dock.widget.annotate_wm_button.pressed.connect(self.start_annotating_wm)
        self.annotation_dock.widget.snap_to_seg.stateChanged.connect(self.plot_curves)

    def init_segmentation_dock(self):
        self.segmentation_dock = SegmentationDockWidget("GM/WM Segmentation")
        self.segmentation_dock.hide()
        self.segmentation_dock.setAllowedAreas(Qt.DockWidgetArea.RightDockWidgetArea)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.segmentation_dock)

        self.segmentation_dock.widget.init_gm_button.pressed.connect(self.initialize_gm)
        self.segmentation_dock.widget.init_wm_button.pressed.connect(self.initialize_wm)
        self.segmentation_dock.widget.reseg_button.pressed.connect(self.resegment_tissue)

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
        self.current_label.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignHCenter)
        self.navigation_toolbar.addWidget(self.current_label)
    
        self.next_button = QPushButton("Next Slide")
        self.next_button.pressed.connect(self.load_next_slide)
        self.navigation_toolbar.addWidget(self.next_button)

    def update_navigation_toolbar(self):
        self.current_label.setText(self.slide_name)
        self.current_label.setFont(QFont("Arial", 24))
        if self.slide_idx != 0:
            #self.previous_button.setText(self.slide_names[self.slide_idx-1])
            self.previous_button.setEnabled(True)
            self.previous_slide_action.setEnabled(True)
        else:
            #self.previous_button.setText("")
            self.previous_button.setEnabled(False)
            self.previous_slide_action.setEnabled(False)
        if self.slide_idx != len(self.slide_names)-1:
            #self.next_button.setText(self.slide_names[self.slide_idx-1])
            self.next_button.setEnabled(True)
            self.next_slide_action.setEnabled(True)
        else:
            #self.next_button.setText("")
            self.next_button.setEnabled(False)
            self.next_slide_action.setEnabled(False)

    def load_slide(self, idx):
        if idx < 0 or idx >= len(self.slide_names):
            dlg = QMessageBox(self)
            dlg.setWindowTitle("Error")
            dlg.setText("Slide index out of bounds")
            button = dlg.exec()
            return

        try:
            self.slide_idx = idx
            self.slide_name = self.slide_names[self.slide_idx]
            self.slide_dir = self.slide_dirs[self.slide_idx]
            self.slide_tb = sana.image.Frame(os.path.join(self.basedir, self.slide_dir, self.FILE_NAMES["Thumbnail"]))
            #self.features = sana.image.Frame(np.load(os.path.join(self.basedir, self.slide_dir, self.FILE_NAMES["Features"])))
            self.train_features = sana.image.Frame(np.load(os.path.join(self.basedir, self.slide_dir, self.FILE_NAMES["Train Features"])))
            self.gm_mask = sana.image.Frame(np.load(os.path.join(self.basedir, self.slide_dir, self.FILE_NAMES["GM Mask"])).astype(np.uint8))
            self.wm_mask = sana.image.Frame(np.load(os.path.join(self.basedir, self.slide_dir, self.FILE_NAMES["WM Mask"])).astype(np.uint8))
            self.tissue_mask = sana.image.Frame(np.load(os.path.join(self.basedir, self.slide_dir, self.FILE_NAMES["Tissue Mask"])).astype(np.uint8))
            self.update_ui_actions(True)
            print(self.slide_tb.size(), self.tissue_mask.size())
            self.tissue_mask.resize(self.slide_tb.size())
            self.gm_mask.resize(self.slide_tb.size())
            self.wm_mask.resize(self.slide_tb.size())
            self.warn_user = self.warnings[idx]
            
            # TODO: simplify polys
            self.gm_polys, self.gm_holes = self.gm_mask.to_polygons()
            self.wm_polys, self.wm_holes = self.wm_mask.to_polygons()
            self.tissue_polys, self.tissue_holes = self.tissue_mask.to_polygons()

            # TODO: update this again when re-training
            self.segmentation_mask = self.tissue_mask.copy()
            self.segmentation_mask.img += self.wm_mask.img
            self.canvas.mask = self.segmentation_mask
            self.interp_gm_polys = [pdnl_sana.interpolate.interp_poly(x) for x in self.gm_polys]
            self.interp_gm_holes = [pdnl_sana.interpolate.interp_poly(x) for x in self.gm_holes]
            self.canvas.gm_polys = self.interp_gm_polys + self.interp_gm_holes
            
        except Exception as e:
            # TODO: save to a logging file/flag for re-annotation
            dlg = QMessageBox(self)
            dlg.setWindowTitle("Error")
            dlg.setText(f"Could not load slide data...\n{e}")
            button = dlg.exec()
            self.slide_name = None
            self.slide_tb = None
            self.features = None
            self.train_features = None
            self.gm_mask = None
            self.wm_mask = None
            self.gm_polys = None
            self.wm_polys = None
            self.tissue_polys = None
            self.tissue_mask = None
            self.update_ui_actions(False)

            return

        self.canvas.annotations = []

        if self.is_annotating:
            self.toggle_is_annotating()
        self.update_navigation_toolbar()
        self.set_source_frame(self.slide_tb)

        if os.path.exists(os.path.join(self.basedir, self.slide_dir, self.FILE_NAMES["Annotation"])):
            annotations = read_annotations(os.path.join(self.basedir, self.slide_dir, self.FILE_NAMES["Annotation"]))
            self.load_annotations(annotations)

        self.showMaximized()
        self.maximize()

    def load_previous_slide(self):
        self.load_slide(self.slide_idx-1)
    def load_next_slide(self):
        self.load_slide(self.slide_idx+1)

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

    def resize_frame(self):
        w = int(round(self.scale_factor * self.plotted_frame.size()[0]))
        h = int(round(self.scale_factor * self.plotted_frame.size()[1]))
        resized_frame = self.plotted_frame.copy()
        resized_frame.resize(sana.geo.Point(w, h), interpolation=cv2.INTER_LINEAR)

        # self.gm_polys = [x * self.scale_factor for x in self.gm_polys]
        # self.wm_polys = [x * self.scale_factor for x in self.wm_polys]
        # self.tissue_polys = [x * self.scale_factor for x in self.tissue_polys]

        self.canvas.scale_factor = self.scale_factor

        self.set_resized_frame(resized_frame)
    
    def plot_curves(self):
        overlay = PIL.Image.fromarray(self.overlaid_frame.img)
        draw = PIL.ImageDraw.Draw(overlay)
        for poly in self.wm_polys:
            poly = poly.astype(int)
            x, y = poly.T
            if self.warn_user:
                draw.polygon(list(zip(x, y)), fill=None, outline="red", width=5)
            else:
                draw.polygon(list(zip(x, y)), fill=None, outline="green", width=5)

        if self.annotation_dock.widget.snap_to_seg.isChecked():
            for poly in self.tissue_polys+self.tissue_holes:
                poly = poly.astype(int)
                x, y = poly.T
                draw.polygon(list(zip(x, y)), fill=None, outline="black", width=5)

        self.set_plotted_frame(sana.image.frame_like(self.source_frame, np.asarray(overlay)))

    def overlay_features(self):
        do_overlay = False
        overlay = sana.image.frame_like(self.source_frame, np.zeros_like(self.source_frame.img))

        if self.show_density_action.isChecked():
            mi = int(self.overlay_dock.widget.density_overlay.min.slider.value()) / 1000
            mx = int(self.overlay_dock.widget.density_overlay.max.slider.value()) / 1000
            mu = np.nanmean(self.features.img[:,:,0])
            sd = np.nanstd(self.features.img[:,:,0])
            rng = 3*sd
            mi = (mu - rng) * (2*mi-1)
            mx = (mu + rng) * (2*mx-1)

            feature = self.features.img[:,:,0]
            overlay.img[:,:,0] = np.rint(255*(np.clip(feature, mi, mx) - mi) / (mx-mi)).astype(np.uint8)
            do_overlay = True

        if self.show_size_action.isChecked():
            mi = int(self.overlay_dock.widget.size_overlay.min.slider.value()) / 1000
            mx = int(self.overlay_dock.widget.size_overlay.max.slider.value()) / 1000
            rng = np.max(self.features.img[:,:,1]) - np.min(self.features.img[:,:,1])
            mu = np.nanmean(self.features.img[:,:,1])
            sd = np.nanstd(self.features.img[:,:,1])
            rng = 3*sd

            mi = (mu - rng) * (2*mi-1)
            mx = (mu + rng) * (2*mx-1)

            feature = self.features.img[:,:,1]
            overlay.img[:,:,1] = 255-np.rint(255*(np.clip(feature, mi, mx) - mi) / (mx-mi)).astype(np.uint8)
            do_overlay = True

        if do_overlay:
            alpha = 0.5
            alpha = sana.image.frame_like(self.source_frame, np.full(self.source_frame.img.shape[:2], alpha)[:,:,None])
            alpha.img[self.tissue_mask.img == 0] = 0

            overlay.img = np.rint((self.source_frame.img * (1.0 - alpha.img) + overlay.img * alpha.img)).astype(np.uint8)
            return self.set_overlaid_frame(overlay)
        else:
            return self.set_overlaid_frame(self.source_frame)
        # TODO: N/A or flag slide button (reason dropdown or text field)
        # TODO: add flip segmetnations button if the GM/WM cluseters were labeled incorrectly
        # TODO: file -> export local annotations to .zip file (save .zip in file dialog)
        # TODO: create cmap dropdown
        # TODO: flag image for re-segmentation if segmentation is bad, either draw and classify using supervised, or re-init GMM


    def set_source_frame(self, frame):
        self.source_frame = frame

        self.aspect_ratio = self.source_frame.size()[1] / self.source_frame.size()[0]

        self.overlay_features()

    def set_overlaid_frame(self, frame):
        self.overlaid_frame = frame

        self.plot_curves()

    def set_plotted_frame(self, frame):
        self.plotted_frame = frame

        self.resize_frame()

    def set_resized_frame(self, frame):
        self.resized_frame = frame

        self.set_current_frame(self.resized_frame)

    def set_current_frame(self, frame):
        self.current_frame = frame
        self.current_image = self.frame_to_qimage(self.current_frame)

        # self.canvas.image = self.current_image
        # self.canvas.adjustSize()

        self.canvas.setPixmap(QPixmap.fromImage(self.current_image))
        self.canvas.adjustSize()

    def open_directory(self):
        options = QFileDialog.Options()
        basedir = QFileDialog.getExistingDirectory(self, caption="Open Preprocessed WSI Archive", options=options)
        return basedir

    def reset_annotations(self):
        self.canvas.reset_annotations()        
        self.save_annotations()
    
    def start_annotating_gm(self):
        self.canvas.current_annotation = Annotation(is_gm=True, is_wm=self.annotation_dock.widget.annotate_adjacent_wm.isChecked())
        self.canvas.gm_width = self.annotation_dock.widget.gm_width_spinbox.value()
        self.toggle_is_annotating(True)
    def start_annotating_wm(self):
        self.canvas.current_annotation = Annotation(is_gm=False, is_wm=True)
        self.toggle_is_annotating(True)

    def initialize_gm(self):
        self.is_segmenting_gm = True
        self.toggle_mouse_tracking(True)
    def initialize_wm(self):
        self.is_segmenting_wm = True
        self.toggle_mouse_tracking(True)

    def zoom_in(self):
        self.set_scale_factor(self.scale_factor * 1.25)
        self.update_scroll_bars(1.25)

        if (self.current_image.width() > 10000) or (self.current_image.height() > 10000):
            self.zoom_in_action.setEnabled(False)
        else:
            self.zoom_in_action.setEnabled(True)
        self.zoom_out_action.setEnabled(True)
                
    def zoom_out(self):
        self.set_scale_factor(self.scale_factor / 1.25)
        self.update_scroll_bars(1/1.25)

        if (self.current_image.width() < 100) or (self.current_image.height() < 100):
            self.zoom_out_action.setEnabled(False)
        else:
            self.zoom_out_action.setEnabled(True)
        self.zoom_in_action.setEnabled(True)

    def reset_zoom(self):
        self.set_scale_factor(1.0)
        self.update_scroll_bars(1)

        self.zoom_in_action.setEnabled(True)
        self.zoom_out_action.setEnabled(True)

    # TODO: spamming ctrl+f hides and shows scroll bars, something funky in the logic, but mostly accurate
    def maximize(self):

        # calculate available height for the canvas
        h = self.DESKTOP_HEIGHT
        h -= self.TITLEBAR_HEIGHT
        if self.scroll_area.horizontalScrollBar().isVisible():
            h -= self.SCROLLBAR_EXTENT
        if self.navigation_toolbar.isVisible():
            h -= self.navigation_toolbar.height()

        # calculate available width for the canvas
        w = self.DESKTOP_WIDTH
        if self.scroll_area.verticalScrollBar().isVisible():
            w -= self.SCROLLBAR_EXTENT
        if self.overlay_dock.isVisible() and not self.overlay_dock.isFloating():
            w -= max([self.overlay_dock.width(), self.annotation_dock.width(), self.segmentation_dock.width()])

        screen_ratio = h / w

        # limited by height
        if self.aspect_ratio/screen_ratio > 1:

            # set the amount of zoom to use all available height
            self.set_scale_factor(h / self.source_frame.size()[1])

            # top left corner to move the window to center it
            px = self.DESKTOP_RECT.width()//2 - self.current_image.width()//2
            py = 0

            # resize to size of screen available
            screen_h = self.DESKTOP_HEIGHT - self.TITLEBAR_HEIGHT
            screen_w = self.current_image.width()
            if self.overlay_dock.isVisible() and not self.overlay_dock.isFloating():
                screen_w += max([self.overlay_dock.width(), self.annotation_dock.width(), self.segmentation_dock.width()])
            if self.scroll_area.verticalScrollBar().isVisible():
                screen_w += self.SCROLLBAR_EXTENT

        # width bigger than the height
        else:
            # set the amount of zoom to use all available width
            self.set_scale_factor(w / self.source_frame.size()[0])

            # top left corner to move the window to center it
            px = 0
            py = self.DESKTOP_RECT.height()//2 - self.current_image.height()//2

            # resize to size of screen available
            screen_h = self.current_image.height()
            if self.navigation_toolbar.isVisible():
                screen_h += self.navigation_toolbar.height()
            if self.scroll_area.horizontalScrollBar().isVisible():
                screen_h += self.SCROLLBAR_EXTENT
            screen_w = self.DESKTOP_WIDTH

        # center the window on the screen
        self.move(QPoint(px, py))

        # resize the window to the image size
        #self.resize(screen_w, screen_h)
        self.showMaximized()
        
        # self.canvas.adjustSize()
        # self.scale_factor = 1.0
        # self.scale_image(1.0)
        pass

    def set_scale_factor(self, scale_factor):
        self.scale_factor = scale_factor
        self.resize_frame()

    def update_scroll_bars(self, factor):
        self.update_scroll_bar(self.scroll_area.horizontalScrollBar(), factor)
        self.update_scroll_bar(self.scroll_area.verticalScrollBar(), factor)
    def update_scroll_bar(self, scroll_bar, factor):
        scroll_bar.setValue(int(factor * scroll_bar.value() + ((factor - 1) * scroll_bar.pageStep() / 2)))

    def create_actions(self):
        self.open_action = QAction("Open...", self, shortcut="Ctrl+O", triggered=self.open_directory)
        self.previous_slide_action = QAction("Open Previous Slide", self, shortcut="Ctrl+<", triggered=self.load_previous_slide, enabled=False)
        self.next_slide_action = QAction("Open Next Slide", self, shortcut="Ctrl+>", triggered=self.load_next_slide, enabled=False)
        self.exit_action = QAction("Exit", self, shortcut="Ctrl+Q", triggered=self.close)
        self.reset_action = QAction("Reset Annotations", self, shortcut="Ctrl+R", triggered=self.reset_annotations, enabled=False)
        self.zoom_in_action = QAction("Zoom In (25%)", self, shortcut="Ctrl+=", enabled=False, triggered=self.zoom_in)
        self.zoom_out_action = QAction("Zoom Out (25%)", self, shortcut="Ctrl+-", enabled=False, triggered=self.zoom_out)
        self.reset_zoom_action = QAction("Reset Zoom", self, shortcut="Ctrl+0", enabled=False, triggered=self.reset_zoom)
        self.maximize_action = QAction("Maximize", self, shortcut="Ctrl+F", enabled=False, triggered=self.maximize)
        self.show_density_action = QAction("Show Density", self, shortcut="Ctrl+1", enabled=False, checkable=True, triggered=self.overlay_features)
        self.show_size_action = QAction("Show Size", self, shortcut="Ctrl+2", enabled=False, checkable=True, triggered=self.overlay_features)

    def create_menus(self):
        self.file_menu = QMenu("File", self)
        self.file_menu.addAction(self.open_action)
        self.file_menu.addSeparator()
        self.file_menu.addAction(self.previous_slide_action)
        self.file_menu.addAction(self.next_slide_action)
        self.file_menu.addSeparator()
        self.file_menu.addAction(self.exit_action)

        self.edit_menu = QMenu("Edit", self)
        self.edit_menu.addAction(self.reset_action)

        self.view_menu = QMenu("View", self)
        #self.view_menu.addAction(self.overlay_dock.toggleViewAction())
        #self.view_menu.addAction(self.show_density_action)
        #self.view_menu.addAction(self.show_size_action)

        self.window_menu = QMenu("Window", self)
        self.window_menu.addAction(self.zoom_in_action)
        self.window_menu.addAction(self.zoom_out_action)
        self.window_menu.addAction(self.reset_zoom_action)
        self.window_menu.addSeparator()
        self.window_menu.addAction(self.maximize_action)

        self.menuBar().addMenu(self.file_menu)
        self.menuBar().addMenu(self.edit_menu)
        self.menuBar().addMenu(self.view_menu)
        self.menuBar().addMenu(self.window_menu)

    def update_ui_actions(self, flag):
        self.scroll_area.setVisible(flag)
        self.zoom_in_action.setEnabled(flag)
        self.zoom_out_action.setEnabled(flag)
        self.reset_zoom_action.setEnabled(flag)
        self.maximize_action.setEnabled(flag)
        self.show_density_action.setEnabled(flag)
        self.show_size_action.setEnabled(flag)
        self.reset_action.setEnabled(flag)
        #self.overlay_dock.show()
        self.annotation_dock.show()
        #self.segmentation_dock.show()
        self.navigation_toolbar.show()

    def scale_image(self, factor):
        self.scale_factor *= factor
        self.canvas.resize(self.scale_factor * self.canvas.pixmap().size())

        self.adjust_scroll_bar(self.scroll_area.horizontalScrollBar(), factor)
        self.adjust_scroll_bar(self.scroll_area.verticalScrollBar(), factor)

        self.zoom_in_action.setEnabled(self.scale_factor < 3.0)
        self.zoom_out_action.setEnabled(self.scale_factor > 0.333)

    def adjust_scroll_bar(self, scroll_bar, factor):
        scroll_bar.setValue(int(factor * scroll_bar.value()
                               + ((factor - 1) * scroll_bar.pageStep() / 2)))

    def resegment_tissue(self):
        if self.init_gm is None or self.init_wm is None:
            return
        
        tissue_size = np.array(self.tissue_mask.img.shape[:2])
        feature_size = np.array(self.train_features.img.shape[:2])
        ds = tissue_size / feature_size

        gm_coord = np.rint(np.array([self.init_gm.x()/ds[0], self.init_gm.y()/ds[1]])).astype(int)
        wm_coord = np.rint(np.array([self.init_wm.x()/ds[0], self.init_wm.y()/ds[1]])).astype(int)

        k = 5
        gm_coords = []
        for j in range(gm_coord[1]-k, gm_coord[1]+k):
            for i in range(gm_coord[0]-k, gm_coord[0]+k):
                gm_coords.append((j,i))
        gm_coords = np.array(gm_coords)
        wm_coords = []
        for j in range(wm_coord[1]-k, wm_coord[1]+k):
            for i in range(wm_coord[0]-k, wm_coord[0]+k):
                wm_coords.append((j,i))
        wm_coords = np.array(wm_coords)

        tissue_mask_down = self.tissue_mask.copy()
        tissue_mask_down.resize(self.train_features.size())
        trn = self.train_features.copy()
        trn.mask(tissue_mask_down)

        # TODO: fix this!! code doesn't exist, need to copy from GMM_segmentation.py
        model = sana.process.train_wm_segmenter(trn, wm_coords=wm_coords, gm_coords=gm_coords)

        self.wm_mask = sana.process.deploy_wm_segmenter(model, self.features)
        self.wm_mask.mask(self.tissue_mask)
        self.wm_polys, self.wm_holes = self.wm_mask.to_polygons()

        self.plot_curves()


class LabeledSlider(QWidget):
    def __init__(self, label, orientation, value):
        super().__init__()

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.label = QLabel(label)
        self.layout.addWidget(self.label)

        self.slider = QSlider(orientation=orientation)
        self.slider.setMinimum(1)
        self.slider.setMaximum(1000)
        self.slider.setSliderPosition(value)
        self.layout.addWidget(self.slider)

class FeatureOverlayWidget(QWidget):
    def __init__(self, label):
        super().__init__()

        self.layout = QHBoxLayout()
        self.setLayout(self.layout)

        self.label = QLabel(label)
        self.layout.addWidget(self.label)

        self.min = LabeledSlider("Min", Qt.Vertical, 1)
        self.layout.addWidget(self.min)

        self.max = LabeledSlider("Max", Qt.Vertical, 1000)
        self.layout.addWidget(self.max)

class OverlayWidget(QWidget):
    def __init__(self):
        super().__init__()

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.density_overlay = FeatureOverlayWidget("Soma Density")
        self.layout.addWidget(self.density_overlay)

        self.size_overlay = FeatureOverlayWidget("Soma Size")
        self.layout.addWidget(self.size_overlay)

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
        self.annotate_adjacent_wm.setEnabled(False)

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
        self.snap_to_seg.setEnabled(False)
        self.layout.addWidget(self.snap_to_seg)

class SegmentationWidget(QWidget):
    def __init__(self):
        super().__init__()

        self.layout = QHBoxLayout()
        self.setLayout(self.layout)

        self.init_gm_button = QPushButton("Initialize GM")
        self.layout.addWidget(self.init_gm_button)

        self.init_wm_button = QPushButton("Initialize WM")
        self.layout.addWidget(self.init_wm_button)

        self.reseg_button = QPushButton("Re-Segment GM/WM")
        self.layout.addWidget(self.reseg_button)

        self.outlier_slider = QSlider()
        self.layout.addWidget(self.outlier_slider)

class OverlayDockWidget(QDockWidget):
    def __init__(self, name):
        super().__init__(name)

        self.widget = OverlayWidget()
        self.setWidget(self.widget)

class AnnotationDockWidget(QDockWidget):
    def __init__(self, name):
        super().__init__(name)

        self.widget = AnnotationWidget()
        self.setWidget(self.widget)

class SegmentationDockWidget(QDockWidget):
    def __init__(self, name):
        super().__init__(name)

        self.widget = SegmentationWidget()
        self.setWidget(self.widget)

# removes unreadable header data from JSON annotation files
# NOTE: these headers come export JSON files from Qupath
def fix_annotations(ifile):

    # load the data as bytes
    fp = open(ifile, 'rb')
    data = fp.read()
    fp.close()

    # find the index of the first annotation in the json
    ind = data.find(b'[\n')
    if ind == -1:
        ind = data.find(b'[]')
        if ind == -1:
            return
        
    # rewrite the data starting at the first annotation
    fp = open(ifile, 'wb')
    fp.write(data[ind:])
    fp.close()
#
# end of fix_annotation

# loads a JSON annotation file into memory
#  -ifile: input JSON file to be read
#  -class_name: if given, only returns annotations with this class
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
        fp = open(ifile, 'r', encoding='utf-8')
        data = json.loads(fp.read())
    else:
        raise Exception

    # load the annotations
    # NOTE: this could be handled by a GeoJSON package?
    annotations = []
    for annotation in data:

        # get the xy coordinates from the geometry of the annotation
        geo = annotation['geometry']

        # get the class name, if exists
        if 'classification' not in annotation['properties']:
            cname = ""
        else:
            cname = annotation['properties']['classification']['name']
        #
        # end of class name reading

        # get the anno name, if exists
        if 'name' not in annotation['properties']:
            aname = ""
        else:
            aname = annotation['properties']['name']
        #
        # end of anno name reading

        # get the attributes dictionary, if exists
        if 'attributes' not in annotation['properties']:
            attributes = {}
        else:
            attributes = annotation['properties']['attributes']
        #
        # end of attributes reading

        # TODO: this should be simplified.
        #        need to actually handle what a MultiPolygon is
        if geo['type'] == 'MultiPolygon':
            coords_list = geo['coordinates']
            x, y = [], []
            for coords in coords_list:
                x += [float(c[0]) for c in coords[0]]
                y += [float(c[1]) for c in coords[0]]
            x = np.array(x)
            y = np.array(y)
        elif geo['type'] == 'Polygon':
            coords = geo['coordinates']
            x = np.array([float(c[0]) for c in coords[0]])
            y = np.array([float(c[1]) for c in coords[0]])
        elif geo['type'] == 'MultiPoint':
            coords = geo['coordinates']
            x = np.array([float(c[0]) for c in coords])
            y = np.array([float(c[1]) for c in coords])
        elif geo['type'] == 'LineString':
            coords = geo['coordinates']
            x = np.array([float(c[0]) for c in coords])
            y = np.array([float(c[1]) for c in coords])            
        else:
            x = np.array([])
            y = np.array([])
        #
        # end of geo type checking
            
        # create and store the annotation object
        annotations.append(
            sana.geo.Annotation(x, y, ifile, cname, aname,
                       attributes=attributes, is_micron=False, level=0))
    #
    # end of annotation loop

    # only return annotations with the given class name
    if not class_name is None:
        annotations = [a for a in annotations if fnmatch.fnmatch(a.class_name, class_name)]
    if not annotation_name is None:
        annotations = [a for a in annotations if fnmatch.fnmatch(a.annotation_name, annotation_name)]

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
    json.dump(json_annos, open(ofile, 'w'), indent=2)
#
# end of write_annotations

def main(argv):
    if len(argv) > 1:
        basedir = argv[1]
    if len(argv) > 2:
        start_idx = int(argv[2])
    else:
        basedir = None
        start_idx = 0
        

    app = QApplication(argv)
    imageViewer = SlideAnnotator(basedir=basedir, start_idx=start_idx, do_ask_name=False)
    imageViewer.showMaximized()
    sys.exit(app.exec())

if __name__ == '__main__':
    main(sys.argv)
