# gpkg_compare_dashboard.py

import os
import tempfile
from collections import OrderedDict

import numpy as np
import pandas as pd
import geopandas as gpd
import fiona

import panel as pn
import holoviews as hv
import hvplot.pandas  # registers hvplot extension
from bokeh.palettes import Category10, Category20_20
from bokeh.io import curdoc as bokeh_curdoc
import colorcet as cc


# --- Extensions & Theme ---
pn.extension("tabulator", design="bootstrap")  # Bootstrap design; light by default
hv.extension("bokeh")

# -------------------------
# ---- Global Styling -----
# -------------------------
ACCENT = "#2C7FB8"
TITLE = "GeoPackage Table Comparator"
SUBTITLE = "Upload multiple GeoPackages, select tables & common columns, and export a publication-ready scatter plot."

template = pn.template.FastListTemplate(
    title=TITLE,
    sidebar_width=600,
    header_background=ACCENT,
    site="",
    main_max_width="1200px",
    theme="default",
    theme_toggle=True,
    #header=[pn.pane.Markdown(f"### {SUBTITLE}")],
)

pn.config.raw_css.append(
    """
.bk.panel-models-markdown h2, .bk.panel-models-markdown h3, .bk.panel-models-markdown h4 { margin-top: 0.2rem !important; }
.bk.bk-input-group label { font-weight: 600; }
.bk.bk-btn { font-weight: 600; }
"""
)

# -------------------------
# ---- Widgets -------------
# -------------------------
file_input = pn.widgets.FileInput(
    name="Upload GeoPackages", accept=".gpkg", multiple=True, width=320
)
clear_btn = pn.widgets.Button(
    name="Clear Uploads", button_type="warning", icon="trash", width=160
)

tables_accordion = pn.Accordion(
    name="Tables by GeoPackage", active=[], sizing_mode="stretch_width"
)

x_select = pn.widgets.Select(name="X-axis (numeric)", options=[], width=280)
y_select = pn.widgets.Select(name="Y-axis (numeric)", options=[], width=280)

alpha_slider = pn.widgets.FloatSlider(
    name="Point alpha", start=0.2, end=1.0, step=0.05, value=1.0, width=280
)
size_slider = pn.widgets.IntSlider(
    name="Point size", start=3, end=100, step=1, value=50, width=280
)
legend_toggle = pn.widgets.Checkbox(name="Show legend", value=True)
grid_toggle = pn.widgets.Checkbox(name="Show grid", value=True)
title_input = pn.widgets.TextInput(
    name="Plot title", placeholder="Optional plot title", width=320
)

# Hidden trigger to force plot updates on dynamic checkbox changes
plot_trigger = pn.widgets.IntInput(name="plot trigger", value=0, visible=False)

# --- Export controls: PNG is default ---
export_format = pn.widgets.RadioButtonGroup(
    name="Export format",
    options=["SVG", "PNG", "PDF"],
    button_type="primary",
    button_style="solid",
    value="PNG",
)
export_dpi = pn.widgets.IntSlider(
    name="DPI (for PNG)", start=100, end=600, step=50, value=300, width=300
)

export_btn = pn.widgets.FileDownload(
    label="Download High-Resolution Figure",
    button_type="success",
    filename="scatter.png",
    width=300,
)

# Diagnostics / status
common_cols_md = pn.pane.Markdown("", sizing_mode="stretch_width")
points_count_md = pn.pane.Markdown("", sizing_mode="stretch_width")
status_alert = pn.pane.Alert(
    "", alert_type="info", visible=False, sizing_mode="stretch_width"
)

# -------------------------
# ---- Data State ----------
# -------------------------
TMPDIR = tempfile.TemporaryDirectory(prefix="gpkg_compare_")
GPKG_DATA = (
    OrderedDict()
)  # {gpkg_name: {"path": path, "layers": {layer_name: DataFrame}}}
_accordion_checkboxes = OrderedDict()  # gpkg_name -> CheckBoxGroup


def make_palette(n):
    if n <= 10:
        pal = list(Category10[10])

    else:
        pal = list(Category20_20) + list(cc.glasbey[: max(0, n - 20)])
    # print(pal)
    return pal[:n]


# -------------------------
# ---- Scheduling helper (robust next tick) ----
# -------------------------
def schedule_next_tick(func, delay_ms: int = 1000):
    doc = None
    try:
        doc = pn.state.curdoc
    except Exception:
        doc = None
    if doc is None:
        try:
            doc = bokeh_curdoc()
        except Exception:
            doc = None
    try:
        if doc and hasattr(doc, "add_next_tick_callback"):
            doc.add_next_tick_callback(lambda: func())
            return
        if doc and hasattr(doc, "add_timeout_callback"):
            doc.add_timeout_callback(lambda: func(), delay_ms)
            return
    except Exception:
        pass
    func()


# -------------------------
# ---- Helpers -------------
# -------------------------
def bytes_to_tempfile(name, data):
    suffix = os.path.splitext(name)[1] if "." in name else ".gpkg"
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=TMPDIR.name)
    tf.write(data)
    tf.flush()
    tf.close()
    return tf.name


def load_gpkg_tables(gpkg_path):
    """Read all layers (tables) and normalize column names to strings."""
    table_map = OrderedDict()
    layers = fiona.listlayers(gpkg_path)
    for layer in layers:
        try:
            gdf = gpd.read_file(gpkg_path, layer=layer)
            df = (
                pd.DataFrame(gdf.drop(columns="geometry"))
                if "geometry" in gdf.columns
                else pd.DataFrame(gdf)
            )
            # --- CRITICAL FIX: standardize column labels to clean strings ---
            df.columns = [str(c).strip() for c in df.columns]
            table_map[layer] = df
        except Exception as e:
            print(f"Warning: could not read layer '{layer}' from {gpkg_path}: {e}")
    return table_map


def gather_selected_tables():
    selected = {}
    for gpkg_name, chk in _accordion_checkboxes.items():
        for layer in chk.value:
            df = GPKG_DATA[gpkg_name]["layers"].get(layer)
            if df is not None and not df.empty:
                selected[(gpkg_name, layer)] = df
    return selected


def _detect_numeric_columns(df: pd.DataFrame):
    """Return columns considered numeric, with a coercion fallback for string numerics."""
    numeric = []
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            numeric.append(str(c))
            continue
        try:
            coerced = pd.to_numeric(s, errors="coerce")
            if coerced.notna().sum() >= max(5, 0.5 * len(coerced)):
                numeric.append(str(c))
        except Exception:
            pass
    return numeric


def numeric_common_columns(selected_tables):
    if not selected_tables:
        return []
    numeric_sets = []
    for _, df in selected_tables.items():
        num_cols = set(_detect_numeric_columns(df))
        numeric_sets.append(num_cols)
    return sorted(
        set.intersection(*numeric_sets) if numeric_sets else set(),
        key=lambda x: x.lower(),
    )


# ------ Race-free updating of axis selectors ------
def update_column_selects(event=None):
    selected = gather_selected_tables()
    commons = numeric_common_columns(selected)  # list[str]
    # print(commons)
    # Preview (for diagnosis)
    # if commons:
    #     common_cols_md.object = "##### Common numeric columns\n" + ", ".join(f"`{c}`" for c in commons)
    #     common_cols_md.visible = True
    # else:
    #     common_cols_md.object = ""
    #     common_cols_md.visible = False

    # Update options immediately
    x_select.options = commons
    y_select.options = commons

    # Set values next tick to avoid Select blank state
    def _set_values():
        if len(commons) > 0:
            # Keep current choice if still valid, else pick first/second
            x_select.value = x_select.value if x_select.value in commons else commons[0]

            y_select.value = y_select.value if y_select.value in commons else commons[1]

            status_alert.visible = False
        else:
            x_select.value = None
            y_select.value = None
            status_alert.object = (
                "No common numeric columns across the selected tables."
            )
            status_alert.alert_type = "warning"
            status_alert.visible = True

    schedule_next_tick(_set_values, delay_ms=1000)


# -------------------------
# ---- Plot Data & Style ---
# -------------------------
def build_plot_df(selected_tables, x_col, y_col):
    rows = []
    for (gpkg_name, layer), df in selected_tables.items():
        # Columns were normalized to strings at load time, so this membership works now
        if x_col in df.columns and y_col in df.columns:
            sub = df[[x_col, y_col]].copy()
            sub = sub.replace([np.inf, -np.inf], np.nan).dropna(subset=[x_col, y_col])
            sub["group"] = f"{gpkg_name}::{layer}"
            rows.append(sub)
    return (
        pd.concat(rows, axis=0, ignore_index=True)
        if rows
        else pd.DataFrame(columns=[x_col, y_col, "group"])
    )


def _consistent_palette(groups):
    pal = make_palette(len(groups))
    return {g: pal[i] for i, g in enumerate(groups)}


def _empty_plot_frame(grid, title):
    base = hv.Points([]).opts(
        width=900,
        height=600,
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="X",
        ylabel="Y",
        show_grid=grid,
        bgcolor="white",
        fontsize={"title": 14, "labels": 13, "xticks": 11, "yticks": 11, "legend": 11},
    )
    txt = hv.Text(0.5, 0.5, "Select tables and X/Y…").opts(
        text_align="center", text_baseline="middle", fontsize=12
    )
    return (base * txt).opts(
        hv.opts.Overlay(title=title or "Scatter Plot", bgcolor="white")
    )


def make_hv_scatter(x, y, alpha, size, legend, grid, title, trigger):
    """Build the scatter plot; bound to widgets so Panel knows dependencies."""
    selected = gather_selected_tables()

    if not selected:
        points_count_md.object = ""
        return _empty_plot_frame(grid, title)

    xcol = x
    ycol = y
    if (not x or not y) or (x == y):
        points_count_md.object = ""
        return _empty_plot_frame(grid, title)

    xcol = str(xcol)
    ycol = str(ycol)

    df = build_plot_df(selected, xcol, ycol)
    if df.empty:
        points_count_md.object = "##### Points plotted: 0"
        return _empty_plot_frame(grid, title or f"{ycol} vs. {xcol}")

    groups = sorted(df["group"].unique().tolist())
    color_map = _consistent_palette(groups)

    overlays = []
    for g in groups:
        sub = df[df["group"] == g]
        sc = (
            sub.hvplot.scatter(
                x=xcol,
                y=ycol,
                color=color_map[g],
                size=size,
                alpha=alpha,
                legend=True,
                hover=True,
                tools=["hover"],
                xlabel=xcol,
                ylabel=ycol,
            )
            .opts(muted_alpha=0.1)
            .relabel(g)
        )
        overlays.append(sc)

    # Update point count diagnostics
    points_count_md.object = (
        f"##### Points plotted: {len(df):,} across {len(groups)} group(s)"
    )

    overlay = hv.Overlay(overlays).opts(
        hv.opts.Scatter(
            show_grid=grid,
            fontsize={
                "title": 14,
                "labels": 13,
                "xticks": 11,
                "yticks": 11,
                "legend": 11,
            },
            padding=0.05,
            default_tools=["pan", "wheel_zoom", "box_zoom", "reset", "save", "hover"],
            active_tools=["wheel_zoom"],
            line_alpha=0.0,
            marker="circle",
            bgcolor="white",
        ),
        hv.opts.Overlay(
            width=900,
            height=600,
            legend_position="top_right",
            show_legend=legend,
            title=title if title else f"{ycol} vs. {xcol}",
            bgcolor="white",
        ),
    )
    return overlay


# Bind to widget params + trigger (ensures updates)
plot_panel = pn.bind(
    make_hv_scatter,
    x=x_select,
    y=y_select,
    alpha=alpha_slider,
    size=size_slider,
    legend=legend_toggle,
    grid=grid_toggle,
    title=title_input,
    trigger=plot_trigger,
)

# Always-rendered plot area (so clipboard works too)
plot_view = pn.pane.HoloViews(
    plot_panel, sizing_mode="stretch_both", min_height=500, css_classes=["plot-area"]
)


# -------------------------
# ---- Export -------------
# -------------------------
def export_figure_callback():
    pn.io.save.save_png(plot_view.object, 'test.png')
    
    # selected = gather_selected_tables()
    # xcol, ycol = x_select.value, y_select.value
    # if not xcol or not ycol:
    #     raise pn.io.FileDownloadCallbackError(
    #         "Nothing to export. Please select tables and X/Y columns."
    #     )
    # xcol = str(xcol)
    # ycol = str(ycol)

    # df = build_plot_df(selected, xcol, ycol)
    # if df.empty:
    #     raise pn.io.FileDownloadCallbackError(
    #         "No data available for the selected X/Y across chosen tables."
    #     )

    # groups = sorted(df["group"].unique().tolist())
    # color_map = _consistent_palette(groups)

    # plt.style.use("seaborn-v0_8-whitegrid")
    # fig, ax = plt.subplots(figsize=(9, 6), dpi=100)
    # fig.patch.set_facecolor("white")
    # ax.set_facecolor("white")

    # for g in groups:
    #     sub = df[df["group"] == g]
    #     ax.scatter(
    #         sub[xcol],
    #         sub[ycol],
    #         s=size_slider.value * 6,
    #         alpha=alpha_slider.value,
    #         c=color_map[g],
    #         edgecolors="none",
    #         label=g,
    #     )

    # ax.set_xlabel(xcol, fontsize=13, color="black")
    # ax.set_ylabel(ycol, fontsize=13, color="black")
    # if title_input.value:
    #     ax.set_title(title_input.value, fontsize=14, pad=10, color="black")

    # if legend_toggle.value:
    #     ax.legend(
    #         loc="center left",
    #         bbox_to_anchor=(1.02, 0.5),
    #         frameon=False,
    #         fontsize=10,
    #         title="Group",
    #     )

    # ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    # ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    # ax.tick_params(labelsize=11, colors="black")
    # if not grid_toggle.value:
    #     ax.grid(False)

    # fmt = export_format.value.lower()
    # buf = io.BytesIO()
    # dpi = export_dpi.value if fmt == "png" else 300
    # fig.tight_layout()
    # fig.savefig(
    #     buf,
    #     format=fmt,
    #     dpi=dpi,
    #     bbox_inches="tight",
    #     facecolor="white",
    #     edgecolor="white",
    # )
    # plt.close(fig)
    # buf.seek(0)
    # return buf.getvalue()


def _update_export_filename(event=None):
    export_btn.filename = f"scatter.{export_format.value.lower()}"


export_btn.callback = export_figure_callback
export_format.param.watch(_update_export_filename, "value")
_update_export_filename()


# -------------------------
# ---- Accordion Build & Watchers ----
# -------------------------
def _watch_table_selection(event):
    update_column_selects()
    plot_trigger.value += 1  # nudge the plot


def rebuild_tables_accordion():
    _accordion_checkboxes.clear()
    tables_accordion.clear()
    for gpkg_name, meta in GPKG_DATA.items():
        layer_names = list(meta["layers"].keys())
        chk = pn.widgets.CheckBoxGroup(
            name=gpkg_name, options=layer_names, inline=False, width=300
        )
        _accordion_checkboxes[gpkg_name] = chk
        chk.param.watch(_watch_table_selection, "value")
        title = f"📦 {gpkg_name} ({len(layer_names)} tables)"
        tables_accordion.append((title, chk))
    tables_accordion.active = [0] if len(tables_accordion.objects) else []


# -------------------------
# ---- Upload/Clear Callbacks ----
# -------------------------
def _handle_uploaded_files(event=None):
    pairs = []
    v = file_input.value
    names = file_input.filename
    if not v:
        return
    if isinstance(v, (list, tuple)):
        for data, name in zip(v, names):
            pairs.append((name, data))
    else:
        pairs.append((names, v))

    added = []
    for name, data in pairs:
        if not name or not data or not name.lower().endswith(".gpkg"):
            continue
        path = bytes_to_tempfile(name, data)
        tables = load_gpkg_tables(path)
        if not tables:
            continue
        base = os.path.basename(name)
        gpkg_key = base
        i = 2
        while gpkg_key in GPKG_DATA:
            gpkg_key = f"{os.path.splitext(base)[0]}_{i}.gpkg"
            i += 1
        GPKG_DATA[gpkg_key] = {"path": path, "layers": tables}
        added.append(gpkg_key)

    if added:
        rebuild_tables_accordion()
        update_column_selects()
        plot_trigger.value += 1
        status_alert.object = f"Loaded {len(added)} GeoPackage(s): {', '.join(added)}"
        status_alert.alert_type = "success"
        status_alert.visible = True


def _clear_uploads(event=None):
    GPKG_DATA.clear()
    rebuild_tables_accordion()
    x_select.options = []
    y_select.options = []
    x_select.value = None
    y_select.value = None
    plot_trigger.value += 1
    status_alert.object = "Cleared all uploads."
    status_alert.alert_type = "info"
    status_alert.visible = True
    common_cols_md.object = ""
    common_cols_md.visible = False
    points_count_md.object = ""


file_input.param.watch(_handle_uploaded_files, "value")
clear_btn.on_click(_clear_uploads)

# -------------------------
# ---- Layout --------------
# -------------------------
sidebar = pn.Column(
    "#### 1) Upload GeoPackages",
    file_input,
    pn.Row(clear_btn),
    pn.layout.Divider(),
    "#### 2) Choose tables",
    tables_accordion,
    pn.layout.Divider(),
    "#### 3) X/Y columns",
    x_select,
    y_select,
    common_cols_md,  # preview of common columns
    points_count_md,  # points count (diagnostic)
    pn.layout.Divider(),
    "#### Plot options",
    title_input,
    alpha_slider,
    size_slider,
    pn.Row(legend_toggle, grid_toggle),
    pn.layout.Divider(),
    "#### Export",
    export_format,
    export_dpi,
    export_btn,
    sizing_mode="stretch_width",
)

main = pn.Column(
    status_alert,
    pn.pane.Markdown("### Scatter Plot", styles={"font-weight": "600"}),
    plot_view,
    sizing_mode="stretch_both",
)

template.sidebar[:] = [sidebar]
template.main[:] = [main]

# Initial build
rebuild_tables_accordion()
template.servable()
