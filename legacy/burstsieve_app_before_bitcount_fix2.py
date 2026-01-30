import os
import sys
import shutil
import csv
import time
import traceback
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import numpy as np
import cv2
from PIL import Image, ImageOps

from PySide6.QtCore import Qt, QThread, Signal, QSize, QTimer
from PySide6.QtGui import QPixmap, QImage, QAction
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QListWidget, QListWidgetItem, QSpinBox,
    QMessageBox, QProgressBar, QSplitter, QCheckBox, QGroupBox,
    QAbstractItemView
)


def log_error_to_file(message: str) -> None:
    try:
        log_path = os.path.join(os.path.expanduser("~"), "burstsieve_error.log")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write("\n" + "="*80 + "\n")
            f.write(message)
            f.write("\n")
    except Exception:
        pass

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png"}  # MVP only, keep it focused


def human_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m}m {s}s"


def is_image_file(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in 
def log_error_to_file(message: str) -> None:
    try:
        log_path = os.path.join(os.path.expanduser("~"), "burstsieve_error.log")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write("\n" + "="*80 + "\n")
            f.write(message)
            f.write("\n")
    except Exception:
        pass

SUPPORTED_EXTS


def list_images(folder: str) -> List[str]:
    files = []
    for root, _, filenames in os.walk(folder):
        for fn in filenames:
            p = os.path.join(root, fn)
            if is_image_file(p):
                files.append(p)
    files.sort()
    return files


def dhash_64(image_pil: Image.Image) -> int:
    """
    dHash (difference hash) 8x8 -> 64-bit integer.
    Robust-ish for near-duplicates, fast enough for MVP.
    """
    img = ImageOps.exif_transpose(image_pil)  # correct orientation where possible
    img = img.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = np.array(img, dtype=np.uint8)
    diff = pixels[:, 1:] > pixels[:, :-1]
    bits = diff.flatten().astype(np.uint8)
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return h


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def sharpness_score_bgr(image_bgr: np.ndarray) -> float:
    """
    Simple sharpness, variance of Laplacian.
    Higher usually means sharper.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def safe_make_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def make_thumbnail(path: str, max_side: int = 220) -> Optional[QPixmap]:
    try:
        img = Image.open(path)
        img = ImageOps.exif_transpose(img)
        img.thumbnail((max_side, max_side))
        if img.mode != "RGB":
            img = img.convert("RGB")
        arr = np.array(img)
        h, w, _ = arr.shape
        qimg = QImage(arr.data, w, h, 3 * w, QImage.Format_RGB888)
        return QPixmap.fromImage(qimg.copy())
    except Exception:
        return None


@dataclass
class PhotoItem:
    path: str
    dh: int
    sharpness: float
    size_bytes: int
    keep: bool = False


@dataclass
class PhotoGroup:
    items: List[PhotoItem] = field(default_factory=list)

    def recommend_keep(self) -> None:
        if not self.items:
            return
        best_idx = int(np.argmax([it.sharpness for it in self.items]))
        for i, it in enumerate(self.items):
            it.keep = (i == best_idx)

    def kept_item(self) -> Optional[PhotoItem]:
        for it in self.items:
            if it.keep:
                return it
        return None

    def duplicates(self) -> List[PhotoItem]:
        return [it for it in self.items if not it.keep]


class ScanWorker(QThread):
    progress = Signal(int, int)  # done, total
    status = Signal(str)
    finished_scan = Signal(list)  # List[PhotoGroup]
    error = Signal(str)

    def __init__(self, folder: str, threshold: int, parent=None):
        super().__init__(parent)
        self.folder = folder
        self.threshold = threshold
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            t0 = time.time()
            paths = list_images(self.folder)
            total = len(paths)
            if total == 0:
                self.finished_scan.emit([])
                return

            self.status.emit(f"Found {total} images, scanning locally, no cloud, no drama.")
            items: List[PhotoItem] = []

            for idx, p in enumerate(paths, start=1):
                if self._stop:
                    self.status.emit("Scan cancelled.")
                    return
                try:
                    pil = Image.open(p)
                    dh = dhash_64(pil)

                    bgr = cv2.imread(p, cv2.IMREAD_COLOR)
                    if bgr is None:
                        raise ValueError("Could not read image")
                    sharp = sharpness_score_bgr(bgr)
                    size_b = os.path.getsize(p)

                    items.append(PhotoItem(path=p, dh=dh, sharpness=sharp, size_bytes=size_b))
                except Exception:
                    pass

                if idx % 5 == 0 or idx == total:
                    self.progress.emit(idx, total)

            if not items:
                self.finished_scan.emit([])
                return

            self.status.emit("Grouping near-duplicates, making order out of chaos.")
            groups: List[PhotoGroup] = []
            for it in items:
                if self._stop:
                    self.status.emit("Grouping cancelled.")
                    return
                placed = False
                for g in groups:
                    rep = g.items[0]
                    if hamming(it.dh, rep.dh) <= self.threshold:
                        g.items.append(it)
                        placed = True
                        break
                if not placed:
                    groups.append(PhotoGroup(items=[it]))

            dup_groups = [g for g in groups if len(g.items) > 1]

            for g in dup_groups:
                g.recommend_keep()

            t1 = time.time()
            self.status.emit(f"Done in {human_time(t1 - t0)}. Duplicate groups found: {len(dup_groups)}.")
            self.finished_scan.emit(dup_groups)

        except Exception:
            self.error.emit(traceback.format_exc())


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BurstSieve, MVP Desktop Duplicate Culler")
        self.resize(1200, 720)

        self.folder: Optional[str] = None
        self.groups: List[PhotoGroup] = []
        self.worker: Optional[ScanWorker] = None

        self.last_moves: List[Tuple[str, str]] = []

        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        root = QVBoxLayout(central)

        top = QHBoxLayout()
        self.btn_choose = QPushButton("Choose Folder")
        self.btn_scan = QPushButton("Scan")
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setEnabled(False)

        top.addWidget(self.btn_choose)
        top.addWidget(self.btn_scan)
        top.addWidget(self.btn_cancel)

        self.threshold_label = QLabel("Similarity threshold (Hamming):")
        self.threshold_spin = QSpinBox()
        self.threshold_spin.setRange(1, 20)
        self.threshold_spin.setValue(8)

        self.threshold_hint = QLabel("Lower = stricter, Higher = looser")
        self.threshold_hint.setStyleSheet("color: #666;")

        top.addSpacing(10)
        top.addWidget(self.threshold_label)
        top.addWidget(self.threshold_spin)
        top.addWidget(self.threshold_hint)
        top.addStretch(1)

        root.addLayout(top)

        status_row = QHBoxLayout()
        self.status = QLabel("Choose a folder to begin.")
        self.progress = QProgressBar()
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        status_row.addWidget(self.status, 3)
        status_row.addWidget(self.progress, 1)
        root.addLayout(status_row)

        splitter = QSplitter(Qt.Horizontal)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self.group_list = QListWidget()
        self.group_list.setSelectionMode(QAbstractItemView.SingleSelection)
        left_layout.addWidget(QLabel("Duplicate groups"))
        left_layout.addWidget(self.group_list, 1)

        actions_box = QGroupBox("Actions")
        actions_layout = QVBoxLayout(actions_box)

        self.btn_quarantine = QPushButton("Move duplicates to Quarantine")
        self.btn_undo = QPushButton("Undo last move")
        self.btn_export = QPushButton("Export CSV report")

        self.btn_quarantine.setEnabled(False)
        self.btn_undo.setEnabled(False)
        self.btn_export.setEnabled(False)

        actions_layout.addWidget(self.btn_quarantine)
        actions_layout.addWidget(self.btn_undo)
        actions_layout.addWidget(self.btn_export)

        left_layout.addWidget(actions_box)
        splitter.addWidget(left_panel)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        right_layout.addWidget(QLabel("Group details"))

        self.detail_info = QLabel("Select a group to preview.")
        self.detail_info.setWordWrap(True)
        right_layout.addWidget(self.detail_info)

        self.thumb_container = QWidget()
        self.thumb_layout = QVBoxLayout(self.thumb_container)
        self.thumb_layout.setAlignment(Qt.AlignTop)

        right_layout.addWidget(self.thumb_container, 1)

        splitter.addWidget(right_panel)
        splitter.setSizes([420, 780])

        root.addWidget(splitter, 1)

        file_menu = self.menuBar().addMenu("File")
        act_quit = QAction("Quit", self)
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_quit)

        self.btn_choose.clicked.connect(self.choose_folder)
        self.btn_scan.clicked.connect(self.start_scan)
        self.btn_cancel.clicked.connect(self.cancel_scan)
        self.group_list.currentRowChanged.connect(self.show_group)
        self.btn_quarantine.clicked.connect(self.move_duplicates_to_quarantine)
        self.btn_undo.clicked.connect(self.undo_last_move)
        self.btn_export.clicked.connect(self.export_csv)

    def choose_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Choose image folder")
        if folder:
            self.folder = folder
            self.status.setText(f"Folder selected: {folder}")
            self.groups = []
            self.group_list.clear()
            self.clear_thumbs()
            self.progress.setValue(0)
            self.btn_quarantine.setEnabled(False)
            self.btn_export.setEnabled(False)

    def start_scan(self):
        if not self.folder:
            QMessageBox.information(self, "No folder", "Choose a folder first.")
            return

        if self.worker and self.worker.isRunning():
            QMessageBox.information(self, "Scan running", "A scan is already running.")
            return

        threshold = int(self.threshold_spin.value())
        self.worker = ScanWorker(self.folder, threshold)
        self.worker.progress.connect(self.on_progress)
        self.worker.status.connect(self.on_status)
        self.worker.finished_scan.connect(self.on_finished_scan)
        self.worker.error.connect(self.on_error)

        self.btn_cancel.setEnabled(True)
        self.btn_scan.setEnabled(False)
        self.btn_choose.setEnabled(False)
        self.progress.setValue(0)
        self.status.setText("Starting scan...")
        self.group_list.clear()
        self.groups = []
        self.clear_thumbs()

        self.worker.start()

    def cancel_scan(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.btn_cancel.setEnabled(False)
            self.status.setText("Cancelling...")

    def on_progress(self, done: int, total: int):
        pct = int((done / max(total, 1)) * 100)
        self.progress.setValue(pct)
        self.progress.setFormat(f"{done}/{total}")

    def on_status(self, msg: str):
        self.status.setText(msg)

    def on_error(self, msg: str):
        log_error_to_file(msg)
        QMessageBox.critical(self, "Error", msg)
        self._scan_cleanup()

    def _scan_cleanup(self):
        self.btn_cancel.setEnabled(False)
        self.btn_scan.setEnabled(True)
        self.btn_choose.setEnabled(True)

    def on_finished_scan(self, groups: list):
        self.groups = groups
        self.group_list.clear()

        if not groups:
            self.status.setText("No duplicate groups found, your shutter discipline is suspiciously perfect.")
            self._scan_cleanup()
            return

        for idx, g in enumerate(groups, start=1):
            kept = g.kept_item()
            total = len(g.items)
            dupe_count = total - 1
            keep_name = os.path.basename(kept.path) if kept else "None"
            item = QListWidgetItem(f"Group {idx} , {total} images , {dupe_count} duplicates , keep: {keep_name}")
            self.group_list.addItem(item)

        self.group_list.setCurrentRow(0)
        self.btn_quarantine.setEnabled(True)
        self.btn_export.setEnabled(True)
        self._scan_cleanup()

    def clear_thumbs(self):
        while self.thumb_layout.count():
            w = self.thumb_layout.takeAt(0).widget()
            if w:
                w.deleteLater()

    def show_group(self, row: int):
        self.clear_thumbs()
        if row < 0 or row >= len(self.groups):
            self.detail_info.setText("Select a group to preview.")
            return

        g = self.groups[row]
        kept = g.kept_item()
        keep_path = kept.path if kept else "None"
        total_size = sum(it.size_bytes for it in g.items)
        dup_size = sum(it.size_bytes for it in g.duplicates())

        self.detail_info.setText(
            f"Images in group: {len(g.items)}\n"
            f"Recommended keep: {os.path.basename(keep_path)}\n"
            f"Group size: {total_size/1e6:.2f} MB , duplicate candidates: {dup_size/1e6:.2f} MB\n"
            f"Tip: tick or untick ‘Keep’ to override recommendation."
        )

        for it in g.items:
            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(6, 6, 6, 6)

            thumb = QLabel()
            pix = make_thumbnail(it.path, 220)
            if pix:
                thumb.setPixmap(pix)
                thumb.setFixedSize(pix.size())
            else:
                thumb.setText("No preview")
                thumb.setFixedSize(QSize(220, 220))

            meta = QLabel(
                f"{os.path.basename(it.path)}\n"
                f"Sharpness: {it.sharpness:.1f}\n"
                f"Size: {it.size_bytes/1e6:.2f} MB"
            )
            meta.setStyleSheet("color: #333;")

            keep_cb = QCheckBox("Keep")
            keep_cb.setChecked(it.keep)

            def make_toggle_callback(item_ref: PhotoItem, group_ref: PhotoGroup, cb_ref: QCheckBox):
                def _toggle(state: int):
                    if state == Qt.Checked:
                        for other in group_ref.items:
                            other.keep = False
                        item_ref.keep = True
                        self.refresh_group_titles()
                        self.show_group(self.group_list.currentRow())
                    else:
                        if group_ref.kept_item() is None:
                            cb_ref.setChecked(True)
                return _toggle

            keep_cb.stateChanged.connect(make_toggle_callback(it, g, keep_cb))

            row_layout.addWidget(thumb)
            row_layout.addWidget(meta, 1)
            row_layout.addWidget(keep_cb)
            self.thumb_layout.addWidget(row_widget)

    def refresh_group_titles(self):
        for idx, g in enumerate(self.groups, start=1):
            kept = g.kept_item()
            total = len(g.items)
            dupe_count = total - 1
            keep_name = os.path.basename(kept.path) if kept else "None"
            self.group_list.item(idx - 1).setText(
                f"Group {idx} , {total} images , {dupe_count} duplicates , keep: {keep_name}"
            )

    def move_duplicates_to_quarantine(self):
        if not self.folder or not self.groups:
            return

        quarantine = os.path.join(self.folder, "_BurstSieve_Quarantine")
        safe_make_dir(quarantine)

        moves: List[Tuple[str, str]] = []
        moved_count = 0
        moved_bytes = 0

        for g in self.groups:
            for it in g.duplicates():
                src = it.path
                if not os.path.exists(src):
                    continue
                base = os.path.basename(src)
                dst = os.path.join(quarantine, base)

                if os.path.exists(dst):
                    name, ext = os.path.splitext(base)
                    k = 1
                    while True:
                        dst = os.path.join(quarantine, f"{name}__{k}{ext}")
                        if not os.path.exists(dst):
                            break
                        k += 1

                try:
                    shutil.move(src, dst)
                    moves.append((dst, src))
                    moved_count += 1
                    moved_bytes += it.size_bytes
                    it.path = dst
                except Exception:
                    pass

        self.last_moves = moves
        self.btn_undo.setEnabled(bool(self.last_moves))

        QMessageBox.information(
            self,
            "Quarantine complete",
            f"Moved {moved_count} images to Quarantine.\nFreed (approx): {moved_bytes/1e6:.2f} MB.\n"
            f"Nothing was deleted, everything can be undone."
        )

        self.refresh_group_titles()
        self.show_group(self.group_list.currentRow())

    def undo_last_move(self):
        if not self.folder or not self.last_moves:
            return

        undone = 0
        for src_now, dst_original in self.last_moves:
            if not os.path.exists(src_now):
                continue
            safe_make_dir(os.path.dirname(dst_original))
            try:
                shutil.move(src_now, dst_original)
                undone += 1
            except Exception:
                pass

        self.last_moves = []
        self.btn_undo.setEnabled(False)

        QMessageBox.information(self, "Undo", f"Undid {undone} moves. Your photos are back where they started.")

        self.status.setText("Undo complete. For a clean state, re-scan the folder.")
        self.groups = []
        self.group_list.clear()
        self.clear_thumbs()
        self.btn_quarantine.setEnabled(False)
        self.btn_export.setEnabled(False)

    def export_csv(self):
        if not self.groups:
            QMessageBox.information(self, "Nothing to export", "Run a scan first.")
            return

        out_path, _ = QFileDialog.getSaveFileName(
            self, "Export report", "burstsieve_report.csv", "CSV Files (*.csv)"
        )
        if not out_path:
            return

        try:
            with open(out_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["group_id", "keep_path", "duplicate_path", "sharpness_keep", "sharpness_duplicate"])
                for gi, g in enumerate(self.groups, start=1):
                    kept = g.kept_item()
                    keep_path = kept.path if kept else ""
                    keep_sharp = kept.sharpness if kept else ""
                    for dup in g.duplicates():
                        w.writerow([gi, keep_path, dup.path, keep_sharp, dup.sharpness])

            QMessageBox.information(self, "Export complete", f"Saved report to:\n{out_path}")
        except Exception as e:
            QMessageBox.critical(self, "Export failed", str(e))


def main():
    try:
        print("BurstSieve: starting Qt app...")
        app = QApplication(sys.argv)
        app.setQuitOnLastWindowClosed(True)

        win = MainWindow()
        win.show()

        def bring_to_front():
            try:
                win.raise_()
                win.activateWindow()
                win.setWindowState(win.windowState() & ~Qt.WindowMinimized | Qt.WindowActive)
                print("BurstSieve: window shown and activated.")
            except Exception as e:
                print("BurstSieve: bring_to_front failed:", e)

        QTimer.singleShot(200, bring_to_front)

        sys.exit(app.exec())

    except Exception as e:
        print("BurstSieve: fatal error during startup:", repr(e))
        raise


if __name__ == "__main__":
    main()
