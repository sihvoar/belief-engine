#!/usr/bin/env python3
"""
Bayes Tree — PyQt6 Desktop Application
Visual evidence tree editor with Monte Carlo simulation and PDF reports.

Copyright (c) 2026 Ari-Pekka Sihvonen
MIT License — see LICENSE file
"""

import sys
import os
import yaml
import math

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QTreeView, QGroupBox, QFormLayout, QLineEdit,
    QDoubleSpinBox, QComboBox, QSpinBox, QPushButton, QLabel,
    QStatusBar, QToolBar, QMenuBar, QMenu, QFileDialog,
    QMessageBox, QProgressBar, QTabWidget, QTableWidget,
    QTableWidgetItem, QHeaderView, QSizePolicy, QAbstractItemView,
)
from PyQt6.QtCore import (
    Qt, QAbstractItemModel, QModelIndex, QThread, pyqtSignal,
    QSettings, QSize,
)
from PyQt6.QtGui import (
    QAction, QIcon, QColor, QFont, QKeySequence,
)

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import matplotlib.ticker as mticker

from bayes_engine import (
    validate_node, run_simulation, NodeResult,
)


# ── Evidence tree data model ─────────────────────────────────────────────────

class TreeNode:
    """Internal node for the evidence tree model."""

    def __init__(self, data=None, parent=None):
        self.parent_node = parent
        self.child_nodes = []
        self.data = data or {
            'node': 'New evidence',
            'lr_min': 1.0,
            'lr_max': 2.0,
            'lr_dist': 'log_uniform',
            'evidence_type': 'for',
        }

    @property
    def name(self):
        return self.data.get('node', '')

    @property
    def evidence_type(self):
        return self.data.get('evidence_type', 'neutral')

    @property
    def is_root(self):
        return self.parent_node is None

    def child(self, row):
        if 0 <= row < len(self.child_nodes):
            return self.child_nodes[row]
        return None

    def child_count(self):
        return len(self.child_nodes)

    def row(self):
        if self.parent_node:
            return self.parent_node.child_nodes.index(self)
        return 0

    def add_child(self, child):
        child.parent_node = self
        self.child_nodes.append(child)

    def insert_child(self, position, child):
        child.parent_node = self
        self.child_nodes.insert(position, child)

    def remove_child(self, position):
        if 0 <= position < len(self.child_nodes):
            self.child_nodes.pop(position)

    def to_dict(self):
        """Convert to YAML-compatible dict."""
        d = dict(self.data)
        if self.child_nodes:
            d['children'] = [c.to_dict() for c in self.child_nodes]
        elif 'children' in d:
            del d['children']
        return d

    @classmethod
    def from_dict(cls, data, parent=None):
        """Build tree from YAML dict."""
        node = cls(data={k: v for k, v in data.items() if k != 'children'},
                   parent=parent)
        for child_data in data.get('children', []):
            child = cls.from_dict(child_data, parent=node)
            node.child_nodes.append(child)
        return node


class EvidenceTreeModel(QAbstractItemModel):
    """Qt model wrapping the evidence tree."""

    def __init__(self, root_node=None, parent=None):
        super().__init__(parent)
        self._root = root_node or TreeNode(data={
            'node': 'Is the hypothesis true?',
            'prior': 0.5,
        })

    @property
    def root(self):
        return self._root

    def set_root(self, node):
        self.beginResetModel()
        self._root = node
        self.endResetModel()

    def index(self, row, column, parent=QModelIndex()):
        if not self.hasIndex(row, column, parent):
            return QModelIndex()
        parent_node = parent.internalPointer() if parent.isValid() else self._root
        child = parent_node.child(row)
        if child:
            return self.createIndex(row, column, child)
        return QModelIndex()

    def parent(self, index):
        if not index.isValid():
            return QModelIndex()
        node = index.internalPointer()
        parent_node = node.parent_node
        if parent_node is None or parent_node is self._root:
            return QModelIndex()
        return self.createIndex(parent_node.row(), 0, parent_node)

    def rowCount(self, parent=QModelIndex()):
        if parent.column() > 0:
            return 0
        node = parent.internalPointer() if parent.isValid() else self._root
        return node.child_count()

    def columnCount(self, parent=QModelIndex()):
        return 3  # Name, Type, LR

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        node = index.internalPointer()
        col = index.column()

        if role == Qt.ItemDataRole.DisplayRole:
            if col == 0:
                return node.name
            elif col == 1:
                return node.evidence_type
            elif col == 2:
                lr_pt = node.data.get('likelihood_ratio')
                if lr_pt is not None:
                    return f'{float(lr_pt):.2f}'
                lo = node.data.get('lr_min', '')
                hi = node.data.get('lr_max', '')
                if lo and hi:
                    return f'[{float(lo):.2f}–{float(hi):.2f}]'
                return ''

        elif role == Qt.ItemDataRole.ForegroundRole:
            if col == 1:
                etype = node.evidence_type
                if etype == 'for':
                    return QColor('#16a34a')
                elif etype == 'against':
                    return QColor('#dc2626')
                return QColor('#6b7280')

        elif role == Qt.ItemDataRole.FontRole:
            if col == 0 and node.is_root:
                f = QFont()
                f.setBold(True)
                return f

        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            return ['Evidence', 'Type', 'LR'][section]
        return None

    def flags(self, index):
        if not index.isValid():
            return Qt.ItemFlag.NoItemFlags
        return Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable

    def add_child_to(self, parent_index, data=None):
        parent_node = parent_index.internalPointer() if parent_index.isValid() else self._root
        pos = parent_node.child_count()
        self.beginInsertRows(parent_index, pos, pos)
        child = TreeNode(data=data, parent=parent_node)
        parent_node.add_child(child)
        self.endInsertRows()
        return self.index(pos, 0, parent_index)

    def remove_node(self, index):
        if not index.isValid():
            return
        node = index.internalPointer()
        if node.is_root:
            return
        parent = node.parent_node
        parent_index = self.parent(index)
        row = node.row()
        self.beginRemoveRows(parent_index, row, row)
        parent.remove_child(row)
        self.endRemoveRows()

    def node_changed(self, index):
        self.dataChanged.emit(
            self.index(index.row(), 0, self.parent(index)),
            self.index(index.row(), 2, self.parent(index)),
        )

    def to_dict(self):
        return self._root.to_dict()


# ── Simulation worker thread ─────────────────────────────────────────────────

class SimulationWorker(QThread):
    progress = pyqtSignal(int, int)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, data, n_sim):
        super().__init__()
        self.data = data
        self.n_sim = n_sim

    def run(self):
        try:
            results = run_simulation(
                self.data, self.n_sim,
                progress_callback=lambda cur, tot: self.progress.emit(cur, tot),
            )
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))


# ── Matplotlib canvas widgets ────────────────────────────────────────────────

class HistogramCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(6, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setMinimumHeight(250)

    def plot(self, posteriors):
        self.ax.clear()
        self.ax.hist(posteriors, bins=30, color='#3b82f6',
                     edgecolor='#1e40af', alpha=0.85)
        self.ax.set_xlabel('Posterior probability')
        self.ax.set_ylabel('Frequency')
        self.ax.set_title('Posterior Distribution')
        self.ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=1))
        self.fig.tight_layout()
        self.draw()


class ImportanceCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(6, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setMinimumHeight(250)

    def plot(self, importance):
        self.ax.clear()
        if not importance:
            self.draw()
            return

        names = []
        deltas = []
        bar_colors = []
        for item in reversed(importance):
            name = item['name']
            if len(name) > 40:
                name = name[:37] + '...'
            names.append(name)
            deltas.append(abs(item['delta']))
            bar_colors.append('#22c55e' if item['delta'] > 0 else '#ef4444')

        y_pos = range(len(names))
        self.ax.barh(y_pos, deltas, color=bar_colors, height=0.6)
        self.ax.set_yticks(y_pos)
        self.ax.set_yticklabels(names, fontsize=8)
        self.ax.set_xlabel('|Δ posterior|')
        self.ax.set_title('Importance Ranking (leave-one-out)')
        self.ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=2))
        self.fig.tight_layout()
        self.draw()


# ── Node editor panel ────────────────────────────────────────────────────────

class NodeEditor(QGroupBox):
    node_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__('Node Editor', parent)
        self._current_node = None
        self._updating = False

        layout = QFormLayout()
        layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText('Evidence description')
        self.name_edit.textChanged.connect(self._on_change)
        layout.addRow('Name:', self.name_edit)

        self.prior_spin = QDoubleSpinBox()
        self.prior_spin.setRange(0.001, 0.999)
        self.prior_spin.setSingleStep(0.05)
        self.prior_spin.setDecimals(3)
        self.prior_spin.valueChanged.connect(self._on_change)
        self.prior_label = QLabel('Prior:')
        layout.addRow(self.prior_label, self.prior_spin)

        self.type_combo = QComboBox()
        self.type_combo.addItems(['for', 'against', 'neutral'])
        self.type_combo.currentTextChanged.connect(self._on_change)
        self.type_label = QLabel('Evidence type:')
        layout.addRow(self.type_label, self.type_combo)

        self.lr_min_spin = QDoubleSpinBox()
        self.lr_min_spin.setRange(0.001, 1000.0)
        self.lr_min_spin.setSingleStep(0.1)
        self.lr_min_spin.setDecimals(3)
        self.lr_min_spin.valueChanged.connect(self._on_change)
        self.lr_min_label = QLabel('LR min:')
        layout.addRow(self.lr_min_label, self.lr_min_spin)

        self.lr_max_spin = QDoubleSpinBox()
        self.lr_max_spin.setRange(0.001, 1000.0)
        self.lr_max_spin.setSingleStep(0.1)
        self.lr_max_spin.setDecimals(3)
        self.lr_max_spin.valueChanged.connect(self._on_change)
        self.lr_max_label = QLabel('LR max:')
        layout.addRow(self.lr_max_label, self.lr_max_spin)

        self.dist_combo = QComboBox()
        self.dist_combo.addItems(['log_uniform', 'uniform', 'beta'])
        self.dist_combo.currentTextChanged.connect(self._on_change)
        self.dist_label = QLabel('Distribution:')
        layout.addRow(self.dist_label, self.dist_combo)

        self.setLayout(layout)
        self._show_fields(root=False)
        self.setEnabled(False)

    def _show_fields(self, root=False):
        self.prior_label.setVisible(root)
        self.prior_spin.setVisible(root)
        self.type_label.setVisible(not root)
        self.type_combo.setVisible(not root)
        self.lr_min_label.setVisible(not root)
        self.lr_min_spin.setVisible(not root)
        self.lr_max_label.setVisible(not root)
        self.lr_max_spin.setVisible(not root)
        self.dist_label.setVisible(not root)
        self.dist_combo.setVisible(not root)

    def set_node(self, node):
        self._updating = True
        self._current_node = node
        self.setEnabled(node is not None)

        if node is None:
            self._updating = False
            return

        is_root = node.is_root
        self._show_fields(root=is_root)

        self.name_edit.setText(node.data.get('node', ''))

        if is_root:
            self.prior_spin.setValue(node.data.get('prior', 0.5))
        else:
            idx = self.type_combo.findText(node.data.get('evidence_type', 'neutral'))
            self.type_combo.setCurrentIndex(max(0, idx))
            self.lr_min_spin.setValue(float(node.data.get('lr_min', 1.0)))
            self.lr_max_spin.setValue(float(node.data.get('lr_max', 1.0)))
            idx = self.dist_combo.findText(node.data.get('lr_dist', 'log_uniform'))
            self.dist_combo.setCurrentIndex(max(0, idx))

        self._updating = False

    def _on_change(self):
        if self._updating or self._current_node is None:
            return
        node = self._current_node
        node.data['node'] = self.name_edit.text()

        if node.is_root:
            node.data['prior'] = self.prior_spin.value()
        else:
            node.data['evidence_type'] = self.type_combo.currentText()
            node.data['lr_min'] = self.lr_min_spin.value()
            node.data['lr_max'] = self.lr_max_spin.value()
            node.data['lr_dist'] = self.dist_combo.currentText()

        self.node_changed.emit()


# ── Results panel ─────────────────────────────────────────────────────────────

class ResultsPanel(QTabWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Summary tab
        self.stats_table = QTableWidget()
        self.stats_table.setColumnCount(2)
        self.stats_table.setHorizontalHeaderLabels(['Metric', 'Value'])
        self.stats_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        self.stats_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.addTab(self.stats_table, 'Summary')

        # Histogram tab
        self.histogram = HistogramCanvas()
        self.addTab(self.histogram, 'Histogram')

        # Sensitivity tab
        self.sens_table = QTableWidget()
        self.sens_table.setColumnCount(2)
        self.sens_table.setHorizontalHeaderLabels(['Threshold', 'Probability'])
        self.sens_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        self.sens_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.addTab(self.sens_table, 'Sensitivity')

        # Importance tab
        self.importance = ImportanceCanvas()
        self.addTab(self.importance, 'Importance')

        self._show_placeholder()

    def _show_placeholder(self):
        self.stats_table.setRowCount(1)
        self.stats_table.setItem(0, 0, QTableWidgetItem('No results yet'))
        self.stats_table.setItem(0, 1, QTableWidgetItem('Run a simulation'))

    def show_results(self, results):
        # Summary stats
        s = results['stats']
        s_lr = results['lr_stats']
        rows = [
            ('Prior', f'{results["prior"]:.1%}'),
            ('Posterior median', f'{s["median"]:.3%}'),
            ('Posterior mean', f'{s["mean"]:.3%}'),
            ('Std deviation', f'{s["std"]:.3%}'),
            ('90% CI', f'[{s["p5"]:.3%} – {s["p95"]:.3%}]'),
            ('Range', f'[{s["min"]:.3%} – {s["max"]:.3%}]'),
            ('Eff. LR median', f'{s_lr["median"]:.4f}'),
            ('Eff. LR 90% CI', f'[{s_lr["p5"]:.4f} – {s_lr["p95"]:.4f}]'),
        ]
        self.stats_table.setRowCount(len(rows))
        for i, (label, value) in enumerate(rows):
            self.stats_table.setItem(i, 0, QTableWidgetItem(label))
            item = QTableWidgetItem(value)
            item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.stats_table.setItem(i, 1, item)

        # Histogram
        self.histogram.plot(results['posteriors'])

        # Sensitivity
        sens = results['sensitivity']
        sens_rows = [
            ('P(posterior < 5%)', f'{sens["p_lt_5"]:.1%}'),
            ('P(posterior < 10%)', f'{sens["p_lt_10"]:.1%}'),
            ('P(posterior > 50%)', f'{sens["p_gt_50"]:.1%}'),
        ]
        self.sens_table.setRowCount(len(sens_rows))
        for i, (label, value) in enumerate(sens_rows):
            self.sens_table.setItem(i, 0, QTableWidgetItem(label))
            item = QTableWidgetItem(value)
            item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.sens_table.setItem(i, 1, item)

        # Importance
        self.importance.plot(results['importance'])


# ── Main window ──────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    MAX_RECENT = 8

    def __init__(self):
        super().__init__()
        self.setWindowTitle('Bayes Tree')
        self.setMinimumSize(1000, 650)
        self.resize(1200, 750)

        self._current_file = None
        self._results = None
        self._worker = None
        self._modified = False
        self._settings = QSettings('BayesTree', 'BayesTreeGUI')

        self._setup_model()
        self._setup_ui()
        self._setup_menus()
        self._setup_toolbar()
        self._setup_connections()
        self._update_title()

    # ── Setup ─────────────────────────────────────────────────────────────

    def _setup_model(self):
        self.tree_model = EvidenceTreeModel()

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: tree + editor
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self.tree_view = QTreeView()
        self.tree_view.setModel(self.tree_model)
        self.tree_view.setHeaderHidden(False)
        self.tree_view.setAnimated(True)
        self.tree_view.setExpandsOnDoubleClick(True)
        self.tree_view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree_view.expandAll()
        self.tree_view.setColumnWidth(0, 250)
        self.tree_view.setColumnWidth(1, 60)
        self.tree_view.setColumnWidth(2, 90)
        left_layout.addWidget(self.tree_view, stretch=3)

        # Node buttons
        btn_layout = QHBoxLayout()
        self.btn_add_child = QPushButton('+ Child')
        self.btn_add_sibling = QPushButton('+ Sibling')
        self.btn_remove = QPushButton('− Remove')
        self.btn_add_child.setToolTip('Add child node to selected')
        self.btn_add_sibling.setToolTip('Add sibling node next to selected')
        self.btn_remove.setToolTip('Remove selected node')
        btn_layout.addWidget(self.btn_add_child)
        btn_layout.addWidget(self.btn_add_sibling)
        btn_layout.addWidget(self.btn_remove)
        left_layout.addLayout(btn_layout)

        self.node_editor = NodeEditor()
        left_layout.addWidget(self.node_editor, stretch=1)

        splitter.addWidget(left)

        # Right: results
        self.results_panel = ResultsPanel()
        splitter.addWidget(self.results_panel)

        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)

        main_layout.addWidget(splitter)

        # Status bar
        self.statusBar().showMessage('Ready')
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMaximumWidth(200)
        self.statusBar().addPermanentWidget(self.progress_bar)

        # Simulation count
        self.sim_count_spin = QSpinBox()
        self.sim_count_spin.setRange(100, 100000)
        self.sim_count_spin.setValue(10000)
        self.sim_count_spin.setSingleStep(1000)
        self.sim_count_spin.setPrefix('Simulations: ')
        self.sim_count_spin.setFixedWidth(180)
        self.statusBar().addPermanentWidget(self.sim_count_spin)

    def _setup_menus(self):
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu('&File')
        self.act_new = file_menu.addAction('&New', self._new_file)
        self.act_new.setShortcut(QKeySequence.StandardKey.New)
        self.act_open = file_menu.addAction('&Open...', self._open_file)
        self.act_open.setShortcut(QKeySequence.StandardKey.Open)

        self.recent_menu = file_menu.addMenu('Open &Recent')
        self._update_recent_menu()

        file_menu.addSeparator()
        self.act_save = file_menu.addAction('&Save', self._save_file)
        self.act_save.setShortcut(QKeySequence.StandardKey.Save)
        self.act_save_as = file_menu.addAction('Save &As...', self._save_file_as)
        self.act_save_as.setShortcut(QKeySequence('Ctrl+Shift+S'))
        file_menu.addSeparator()
        file_menu.addAction('&Quit', self.close).setShortcut(QKeySequence.StandardKey.Quit)

        # Edit menu
        edit_menu = menubar.addMenu('&Edit')
        edit_menu.addAction('Add &Child Node', self._add_child).setShortcut(QKeySequence('Ctrl+D'))
        edit_menu.addAction('Add &Sibling Node', self._add_sibling).setShortcut(QKeySequence('Ctrl+Shift+D'))
        edit_menu.addSeparator()
        edit_menu.addAction('&Delete Node', self._remove_node).setShortcut(QKeySequence.StandardKey.Delete)

        # Run menu
        run_menu = menubar.addMenu('&Run')
        self.act_simulate = run_menu.addAction('&Simulate', self._run_simulation)
        self.act_simulate.setShortcut(QKeySequence('Ctrl+R'))

        # Report menu
        report_menu = menubar.addMenu('R&eport')
        self.act_report = report_menu.addAction('Generate &PDF Report...', self._generate_report)
        self.act_report.setShortcut(QKeySequence('Ctrl+P'))
        self.act_report.setEnabled(False)

        # Help menu
        help_menu = menubar.addMenu('&Help')
        help_menu.addAction('&About', self._show_about)
        help_menu.addAction('LR &Guide', self._show_lr_guide)

    def _setup_toolbar(self):
        toolbar = self.addToolBar('Main')
        toolbar.setMovable(False)
        toolbar.setIconSize(QSize(16, 16))

        toolbar.addAction('📄 New', self._new_file)
        toolbar.addAction('📂 Open', self._open_file)
        toolbar.addAction('💾 Save', self._save_file)
        toolbar.addSeparator()
        toolbar.addAction('▶ Simulate', self._run_simulation)
        toolbar.addAction('📊 PDF Report', self._generate_report)
        toolbar.addSeparator()
        toolbar.addAction('➕ Add Child', self._add_child)
        toolbar.addAction('➖ Remove', self._remove_node)

    def _setup_connections(self):
        self.tree_view.selectionModel().currentChanged.connect(self._on_selection)
        self.tree_view.customContextMenuRequested.connect(self._context_menu)
        self.node_editor.node_changed.connect(self._on_node_edited)
        self.btn_add_child.clicked.connect(self._add_child)
        self.btn_add_sibling.clicked.connect(self._add_sibling)
        self.btn_remove.clicked.connect(self._remove_node)

    # ── Tree operations ───────────────────────────────────────────────────

    def _on_selection(self, current, previous):
        if current.isValid():
            node = current.internalPointer()
            self.node_editor.set_node(node)
        else:
            self.node_editor.set_node(None)

    def _on_node_edited(self):
        current = self.tree_view.currentIndex()
        if current.isValid():
            self.tree_model.node_changed(current)
        self._set_modified(True)

    def _add_child(self):
        current = self.tree_view.currentIndex()
        parent = current if current.isValid() else QModelIndex()
        new_idx = self.tree_model.add_child_to(parent)
        self.tree_view.expand(parent)
        self.tree_view.setCurrentIndex(new_idx)
        self._set_modified(True)

    def _add_sibling(self):
        current = self.tree_view.currentIndex()
        if not current.isValid():
            return
        node = current.internalPointer()
        if node.is_root:
            self._add_child()
            return
        parent_index = self.tree_model.parent(current)
        new_idx = self.tree_model.add_child_to(parent_index)
        self.tree_view.setCurrentIndex(new_idx)
        self._set_modified(True)

    def _remove_node(self):
        current = self.tree_view.currentIndex()
        if not current.isValid():
            return
        node = current.internalPointer()
        if node.is_root:
            QMessageBox.warning(self, 'Cannot delete',
                                'Cannot delete the root node.')
            return
        reply = QMessageBox.question(
            self, 'Delete node',
            f'Delete "{node.name}" and all its children?',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.tree_model.remove_node(current)
            self._set_modified(True)

    def _context_menu(self, pos):
        index = self.tree_view.indexAt(pos)
        menu = QMenu(self)
        menu.addAction('Add Child', self._add_child)
        if index.isValid():
            node = index.internalPointer()
            if not node.is_root:
                menu.addAction('Add Sibling', self._add_sibling)
                menu.addSeparator()
                menu.addAction('Delete', self._remove_node)
        menu.exec(self.tree_view.viewport().mapToGlobal(pos))

    # ── File operations ───────────────────────────────────────────────────

    def _new_file(self):
        if not self._check_save():
            return
        self._current_file = None
        self._results = None
        self.tree_model.set_root(TreeNode(data={
            'node': 'Is the hypothesis true?',
            'prior': 0.5,
        }))
        self.tree_view.expandAll()
        self.results_panel._show_placeholder()
        self.act_report.setEnabled(False)
        self._set_modified(False)
        self.statusBar().showMessage('New tree created')

    def _open_file(self):
        if not self._check_save():
            return
        path, _ = QFileDialog.getOpenFileName(
            self, 'Open YAML', '',
            'YAML files (*.yaml *.yml);;All files (*)',
        )
        if path:
            self._load_file(path)

    def _load_file(self, path):
        try:
            with open(path, encoding='utf-8') as f:
                data = yaml.safe_load(f)
            root = TreeNode.from_dict(data)
            self.tree_model.set_root(root)
            self.tree_view.expandAll()
            self._current_file = path
            self._results = None
            self.results_panel._show_placeholder()
            self.act_report.setEnabled(False)
            self._set_modified(False)
            self._add_recent(path)
            self.statusBar().showMessage(f'Loaded: {os.path.basename(path)}')

            # Show validation warnings
            warnings = validate_node(data)
            if warnings:
                msg = 'Validation warnings:\n\n' + '\n\n'.join(warnings)
                QMessageBox.warning(self, 'Validation Warnings', msg)
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to load file:\n{e}')

    def _save_file(self):
        if self._current_file:
            self._write_file(self._current_file)
        else:
            self._save_file_as()

    def _save_file_as(self):
        path, _ = QFileDialog.getSaveFileName(
            self, 'Save YAML', '',
            'YAML files (*.yaml *.yml);;All files (*)',
        )
        if path:
            self._write_file(path)

    def _write_file(self, path):
        try:
            data = self.tree_model.to_dict()
            with open(path, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, default_flow_style=False,
                          allow_unicode=True, sort_keys=False)
            self._current_file = path
            self._set_modified(False)
            self._add_recent(path)
            self.statusBar().showMessage(f'Saved: {os.path.basename(path)}')
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to save file:\n{e}')

    def _check_save(self):
        if not self._modified:
            return True
        reply = QMessageBox.question(
            self, 'Unsaved changes',
            'Save changes before continuing?',
            QMessageBox.StandardButton.Save |
            QMessageBox.StandardButton.Discard |
            QMessageBox.StandardButton.Cancel,
        )
        if reply == QMessageBox.StandardButton.Save:
            self._save_file()
            return not self._modified
        elif reply == QMessageBox.StandardButton.Discard:
            return True
        return False

    def _set_modified(self, modified):
        self._modified = modified
        self._update_title()

    def _update_title(self):
        name = os.path.basename(self._current_file) if self._current_file else 'Untitled'
        mod = ' •' if self._modified else ''
        self.setWindowTitle(f'{name}{mod} — Bayes Tree')

    # ── Recent files ──────────────────────────────────────────────────────

    def _add_recent(self, path):
        recent = self._settings.value('recent_files', [], type=list)
        if path in recent:
            recent.remove(path)
        recent.insert(0, path)
        self._settings.setValue('recent_files', recent[:self.MAX_RECENT])
        self._update_recent_menu()

    def _update_recent_menu(self):
        self.recent_menu.clear()
        recent = self._settings.value('recent_files', [], type=list)
        for path in recent:
            if os.path.exists(path):
                act = self.recent_menu.addAction(os.path.basename(path))
                act.setData(path)
                act.triggered.connect(lambda checked, p=path: self._load_file(p))
        if not recent:
            self.recent_menu.addAction('(no recent files)').setEnabled(False)

    # ── Simulation ────────────────────────────────────────────────────────

    def _run_simulation(self):
        if self._worker and self._worker.isRunning():
            QMessageBox.information(self, 'Busy', 'Simulation already running.')
            return

        data = self.tree_model.to_dict()
        n_sim = self.sim_count_spin.value()

        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(n_sim)
        self.progress_bar.setValue(0)
        self.act_simulate.setEnabled(False)
        self.statusBar().showMessage(f'Simulating ({n_sim:,} runs)...')

        self._worker = SimulationWorker(data, n_sim)
        self._worker.progress.connect(self._on_sim_progress)
        self._worker.finished.connect(self._on_sim_finished)
        self._worker.error.connect(self._on_sim_error)
        self._worker.start()

    def _on_sim_progress(self, current, total):
        self.progress_bar.setValue(current)

    def _on_sim_finished(self, results):
        self._results = results
        self.progress_bar.setVisible(False)
        self.act_simulate.setEnabled(True)
        self.act_report.setEnabled(True)
        self.results_panel.show_results(results)
        self.statusBar().showMessage(
            f'Done — median: {results["stats"]["median"]:.3%}  '
            f'90% CI: [{results["stats"]["p5"]:.3%}–{results["stats"]["p95"]:.3%}]'
        )

        if results.get('warnings'):
            msg = 'Validation warnings:\n\n' + '\n\n'.join(results['warnings'])
            QMessageBox.warning(self, 'Warnings', msg)

    def _on_sim_error(self, error_msg):
        self.progress_bar.setVisible(False)
        self.act_simulate.setEnabled(True)
        QMessageBox.critical(self, 'Simulation Error', error_msg)
        self.statusBar().showMessage('Simulation failed')

    # ── Report ────────────────────────────────────────────────────────────

    def _generate_report(self):
        if not self._results:
            QMessageBox.information(self, 'No results',
                                    'Run a simulation first.')
            return

        default_name = 'report.pdf'
        if self._current_file:
            base = os.path.splitext(os.path.basename(self._current_file))[0]
            default_name = f'{base}_report.pdf'

        path, _ = QFileDialog.getSaveFileName(
            self, 'Save PDF Report', default_name,
            'PDF files (*.pdf);;All files (*)',
        )
        if not path:
            return

        try:
            from report_generator import generate_report
            fname = self._current_file or 'Untitled'
            generate_report(self._results, fname, path)
            self.statusBar().showMessage(f'Report saved: {os.path.basename(path)}')
            QMessageBox.information(self, 'Report Generated',
                                    f'PDF saved to:\n{path}')
        except Exception as e:
            QMessageBox.critical(self, 'Report Error',
                                 f'Failed to generate report:\n{e}')

    # ── Help dialogs ──────────────────────────────────────────────────────

    def _show_about(self):
        QMessageBox.about(self, 'About Bayes Tree',
            '<h2>Bayes Tree</h2>'
            '<p>Visual Bayesian evidence tree editor with Monte Carlo simulation.</p>'
            '<p>Structure arguments as likelihood-ratio evidence trees and '
            'let Monte Carlo tell you what to believe.</p>'
            '<p>© 2026 Ari-Pekka Sihvonen — MIT License</p>'
        )

    def _show_lr_guide(self):
        QMessageBox.information(self, 'Likelihood Ratio Guide',
            '<h3>How to Think About Likelihood Ratios</h3>'
            '<table border="1" cellpadding="4">'
            '<tr><th>LR</th><th>Meaning</th><th>Example</th></tr>'
            '<tr><td>10+</td><td>Strong support</td><td>DNA match</td></tr>'
            '<tr><td>2–5</td><td>Moderate support</td><td>Witness nearby</td></tr>'
            '<tr><td>1</td><td>Neutral</td><td>Irrelevant info</td></tr>'
            '<tr><td>0.2–0.5</td><td>Moderate counter</td><td>Single alibi</td></tr>'
            '<tr><td>&lt;0.1</td><td>Strong counter</td><td>Video alibi</td></tr>'
            '</table>'
        )

    # ── Close event ───────────────────────────────────────────────────────

    def closeEvent(self, event):
        if self._check_save():
            event.accept()
        else:
            event.ignore()


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    app.setApplicationName('Bayes Tree')
    app.setOrganizationName('BayesTree')

    # Apply a clean style
    app.setStyle('Fusion')

    window = MainWindow()

    # Load file from command line argument
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        window._load_file(sys.argv[1])

    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
