
import io
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, Optional, List, Tuple
import numpy as np
import pandas as pd
import panel as pn
import holoviews as hv
import geopandas as gpd
try:
    import fiona
except Exception:
    fiona = None
import pydeck as pdk
try:
    from pydeck.data_utils.viewport_helpers import compute_view
except Exception:
    compute_view = None

pn.extension('filedropper', 'tabulator', 'deckgl')
hv.extension('bokeh')

# ======================================
# Widgets
# ======================================
file_dropper = pn.widgets.FileDropper(
    name="Upload GeoPackage(s) (.gpkg)",
    #accepted_filetypes=['.gpkg'],
    multiple=True,  # <-- allow multiple
    max_file_size="500MB",
    layout="integrated",
    sizing_mode="stretch_width"
)

# Scatter controls (single-file workflow, keeps backward compatibility)
layer_select = pn.widgets.Select(name="Table (layer) for Scatter", options=[], disabled=True)
x_select = pn.widgets.Select(name="X axis", options=[], disabled=True)
y_select = pn.widgets.Select(name="Y axis", options=[], disabled=True)

# Map layer selection (independent of scatter unless sync enabled)
map_layer_select = pn.widgets.Select(name="Map layer", options=[], disabled=True)
sync_map_layer = pn.widgets.Checkbox(name="Sync map layer with scatter layer", value=False)

# Map symbology widgets
color_by = pn.widgets.Select(name="Color by", options=[], disabled=True)
color_mode = pn.widgets.RadioButtonGroup(name="Color mode", value='auto', options=['auto', 'continuous', 'categorical'])
palette_select = pn.widgets.Select(
    name="Palette",
    options=['Viridis', 'Plasma', 'Blue-Red', 'Category10'],
    value='Viridis'
)
opacity_slider = pn.widgets.FloatSlider(name="Opacity", start=0.1, end=1.0, step=0.05, value=0.8)

# Smaller dots: default 3px, range 1–15
size_by = pn.widgets.Select(name="Size by (numeric)", options=[None], value=None, disabled=True)
size_scale = pn.widgets.IntSlider(name="Point/Line size (px)", start=1, end=15, step=1, value=3)

basemap_style = pn.widgets.Select(
    name="Basemap style",
    options=['light', 'dark', 'road', 'satellite', 'light_no_labels', 'dark_no_labels'],
    value='light'
)
show_outlines = pn.widgets.Checkbox(name="Show outlines", value=True)

# Map mode / heatmap & hexagon controls
map_mode = pn.widgets.RadioButtonGroup(name="Map mode", options=['GeoJSON', 'Heatmap', 'Hexagon'], value='GeoJSON')
weight_by = pn.widgets.Select(name="Weight by (numeric)", options=[None], value=None, disabled=True)
heat_radius = pn.widgets.IntSlider(name="Heat radius (px)", start=5, end=100, step=1, value=25)
heat_intensity = pn.widgets.FloatSlider(name="Heat intensity", start=0.1, end=5.0, step=0.1, value=1.0)
hex_radius = pn.widgets.IntSlider(name="Hex radius (meters)", start=50, end=5000, step=50, value=500)
hex_extruded = pn.widgets.Checkbox(name="Hex extruded", value=True)
hex_elev_scale = pn.widgets.IntSlider(name="Hex elevation scale", start=1, end=100, step=1, value=20)

status_alert = pn.pane.Alert(
    "Drag & drop one or more .gpkg files above to begin.",
    alert_type="primary",
    sizing_mode="stretch_width",
    margin=(5, 5)
)

# Warning fix: stretch_both + min_height (no fixed height)
deck_pane = pn.pane.DeckGL(object=None, sizing_mode="stretch_both", min_height=520, throttle={'view': 200, 'hover': 200})

# Legend (auto-updates)
legend_pane = pn.pane.HTML(sizing_mode="stretch_width", margin=(10, 0))

# ======================================
# NEW: Multi-GPKG Compare/Overlay widgets
# ======================================
overlay_status = pn.pane.Alert("Upload multiple .gpkg files to compare via spatial overlays.", alert_type="primary")

# Base source (file + layer)
over_base_file = pn.widgets.Select(name="Base file", options=[], disabled=True)
over_base_layer = pn.widgets.Select(name="Base layer (polygons)", options=[], disabled=True)

# Overlay selections (multi)
over_layers = pn.widgets.MultiSelect(name="Overlay layers (file::layer)", options=[], size=8, disabled=True)

# Operation / metrics
over_metric = pn.widgets.RadioButtonGroup(name="Overlay metric", options=[
    'counts (points-in-polygons)',
    'area_intersection (polygons)'
], value='counts (points-in-polygons)')

# Color-by for base result after computing metrics
compare_color_by = pn.widgets.Select(name="Color base by metric", options=[], disabled=True)

# Map styling for compare tab
compare_palette = pn.widgets.Select(name="Compare palette", options=['Viridis', 'Plasma', 'Blue-Red', 'Category10'], value='Viridis')
compare_opacity = pn.widgets.FloatSlider(name="Base fill opacity", start=0.1, end=1.0, step=0.05, value=0.6)

compute_btn = pn.widgets.Button(name="Compute overlay metrics", button_type="primary")

compare_deck = pn.pane.DeckGL(object=None, sizing_mode="stretch_both", min_height=520)
compare_legend = pn.pane.HTML(sizing_mode="stretch_width")
compare_table = pn.widgets.Tabulator(pd.DataFrame(), height=250)

# ======================================
# State
# ======================================
_tmp_dir = Path(tempfile.mkdtemp(prefix="gpkg_panel_"))
_current_gpkg_path: Optional[Path] = None
_gdf_cache: Dict[Tuple[str, str], gpd.GeoDataFrame] = {}  # (file_key, layer) -> GeoDataFrame

# NEW: hold all uploaded files and their layers
_all_files: Dict[str, Path] = {}
_layers_by_file: Dict[str, List[str]] = {}

# Store last computed compare result for recoloring
_compare_base_gdf: Optional[gpd.GeoDataFrame] = None

# ======================================
# Helpers
# ======================================

def _save_uploaded_files(value: dict) -> List[Path]:
    paths = []
    if not value:
        return paths
    for fname, content in list(value.items()):
        data = content.encode('utf-8') if isinstance(content, str) else content
        path = _tmp_dir / Path(fname).name
        # If duplicate filename, uniquify
        i = 1
        base, ext = os.path.splitext(path.name)
        out = path
        while out.exists():
            out = _tmp_dir / f"{base}_{i}{ext}"
            i += 1
        out.write_bytes(data)
        paths.append(out)
    return paths


def _list_layers(path: Path) -> List[str]:
    try:
        layers_df = gpd.list_layers(str(path))
        names = layers_df["name"].tolist()
        if names:
            return names
    except Exception:
        pass
    if fiona is None:
        raise RuntimeError("Could not list layers via geopandas and Fiona is not available.")
    return list(fiona.listlayers(str(path)))


def _read_layer(file_key: str, layer: str) -> gpd.GeoDataFrame:
    key = (file_key, layer)
    if key in _gdf_cache:
        return _gdf_cache[key]
    gdf = gpd.read_file(str(_all_files[file_key]), layer=layer)
    _gdf_cache[key] = gdf
    return gdf


def _get_geometry_info(gdf: gpd.GeoDataFrame) -> Tuple[bool, Optional[str]]:
    if "geometry" not in gdf or gdf.geometry.is_empty.all():
        return False, None
    geom_types = gdf.geometry.geom_type.dropna().unique().tolist()
    for t in ['Point', 'MultiPoint', 'LineString', 'MultiLineString', 'Polygon', 'MultiPolygon']:
        if any(gt == t for gt in geom_types):
            return True, t
    return True, geom_types[0] if geom_types else None


def _numeric_columns(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(include=[np.number]).columns.tolist()


def _to_wgs84(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    try:
        if gdf.crs is None:
            return gdf.set_crs(4326, allow_override=True)
        return gdf.to_crs(4326)
    except Exception:
        return gdf  # best effort


def _compute_view_state(gdf: gpd.GeoDataFrame) -> pdk.ViewState:
    try:
        if compute_view is not None:
            vs = compute_view(gdf)
            return pdk.ViewState(**vs)
    except Exception:
        pass
    minx, miny, maxx, maxy = gdf.total_bounds
    lon = float(minx + (maxx - minx)/2) if np.isfinite(minx) and np.isfinite(maxx) else 0
    lat = float(miny + (maxy - miny)/2) if np.isfinite(miny) and np.isfinite(maxy) else 0
    return pdk.ViewState(longitude=lon, latitude=lat, zoom=3)

# palettes (RGB)
_PALETTES = {
    'Viridis': [
        (68, 1, 84), (71, 44, 122), (59, 81, 139), (44, 113, 142),
        (33, 144, 141), (39, 173, 129), (92, 200, 99), (170, 220, 50), (253, 231, 37)
    ],
    'Plasma': [
        (13, 8, 135), (84, 2, 163), (139, 10, 165), (185, 50, 137),
        (219, 92, 104), (244, 136, 72), (254, 188, 44), (240, 249, 33)
    ],
    'Blue-Red': [(49, 130, 189), (107, 174, 214), (189, 0, 38)],
    'Category10': [
        (31,119,180), (255,127,14), (44,160,44), (214,39,40), (148,103,189),
        (140,86,75), (227,119,194), (127,127,127), (188,189,34), (23,190,207)
    ]
}


def _interp_palette(palette_name: str, steps: int = 256) -> np.ndarray:
    base = np.array(_PALETTES[palette_name], dtype=float)
    if base.shape[0] == 1:
        return np.repeat(base, steps, axis=0)
    x = np.linspace(0, 1, base.shape[0])
    xi = np.linspace(0, 1, steps)
    r = np.interp(xi, x, base[:,0]); g = np.interp(xi, x, base[:,1]); b = np.interp(xi, x, base[:,2])
    return np.stack([r, g, b], axis=1).astype(int)


def _colorize_series(s: pd.Series, mode: str, palette: str, opacity: float) -> List[List[int]]:
    a = int(255 * np.clip(opacity, 0, 1))
    if mode == 'categorical' or (mode == 'auto' and not np.issubdtype(s.dtype, np.number)):
        cats = pd.Categorical(s.astype(str))
        pal = np.array(_PALETTES['Category10'])  # categorical palette
        colors = pal[(cats.codes % len(pal))]
        return [[int(r), int(g), int(b), a] for r, g, b in colors]
    vals = pd.to_numeric(s, errors='coerce')
    finite = vals.replace([np.inf, -np.inf], np.nan)
    vmin, vmax = finite.min(), finite.max()
    if pd.isna(vmin) or pd.isna(vmax) or vmin == vmax:
        rgb = _PALETTES[palette][0]
        return [[int(rgb[0]), int(rgb[1]), int(rgb[2]), a]] * len(s)
    ramp = _interp_palette(palette, 256)
    t = np.clip(((finite - vmin) / (vmax - vmin)).fillna(0), 0, 1)
    idx = (t * 255).round().astype(int)
    colors = ramp[idx.values]
    return [[int(r), int(g), int(b), a] for r, g, b in colors]


def _size_from_series(s: pd.Series, scale_px: int) -> List[float]:
    vals = pd.to_numeric(s, errors='coerce')
    vmin, vmax = vals.min(), vals.max()
    if pd.isna(vmin) or pd.isna(vmax) or vmin == vmax:
        return [max(1, scale_px)] * len(s)
    t = np.clip((vals - vmin) / (vmax - vmin), 0, 1)
    return (1 + t * (scale_px - 1)).tolist()


def _geom_to_points(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """Return DataFrame with 'lon','lat' for Heatmap/Hexagon; use representative point for non-points."""
    gdf = _to_wgs84(gdf)
    pts = gdf.geometry.representative_point()
    return pd.DataFrame({'lon': pts.x.values, 'lat': pts.y.values})


def _get_series_stats(s: pd.Series) -> Tuple[Optional[float], Optional[float]]:
    vals = pd.to_numeric(s, errors='coerce')
    finite = vals.replace([np.inf, -np.inf], np.nan).dropna()
    return (finite.min(), finite.max()) if not finite.empty else (None, None)

# ======================================
# Build Deck & Legend (existing map tab)
# ======================================

def _build_deck(gdf_in: gpd.GeoDataFrame,
                mode: str,
                color_col: Optional[str],
                color_mode_val: str,
                palette: str,
                opacity: float,
                size_col: Optional[str],
                size_scale_px: int,
                basemap: str,
                outlines: bool,
                weight_col: Optional[str],
                heat_radius_px: int,
                heat_intensity_val: float,
                hex_radius_m: int,
                hex_extruded_val: bool,
                hex_elev_scale_val: int) -> Optional[pdk.Deck]:
    if gdf_in is None or gdf_in.empty:
        return None
    gdf = _to_wgs84(gdf_in.copy())
    has_geom, _ = _get_geometry_info(gdf)
    if not has_geom:
        return None
    view_state = _compute_view_state(gdf)

    if mode == 'GeoJSON':
        # Color & size columns
        if color_col and color_col in gdf.columns:
            gdf["__color__"] = _colorize_series(gdf[color_col], color_mode_val, palette, opacity)
        else:
            base = _PALETTES[palette][0]
            gdf["__color__"] = [[base[0], base[1], base[2], int(255*opacity)]] * len(gdf)
        if size_col and size_col in gdf.columns:
            gdf["__size__"] = _size_from_series(gdf[size_col], size_scale_px)
        else:
            gdf["__size__"] = [size_scale_px] * len(gdf)

        layer = pdk.Layer(
            "GeoJsonLayer",
            data=gdf,
            pickable=True,
            auto_highlight=True,
            stroked=True,
            filled=True,
            get_fill_color="__color__",
            get_line_color="__color__",
            get_point_radius="__size__",
            get_line_width="__size__",
            point_radius_units="pixels",
            line_width_units="pixels",
            line_width_min_pixels=0.5,
            line_width_max_pixels=100,
            opacity=float(opacity),
        )
        if not outlines:
            layer.kwargs["get_line_color"] = [0, 0, 0, 0]
        tooltip_fields = [{"text": f"{c}: {{{{{c}}}}}"} for c in list(gdf.columns)[:10] if c != "geometry"]
        tooltip = {"html": "<br/>".join([f["text"] for f in tooltip_fields])} if tooltip_fields else True
        return pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            map_provider="carto",
            map_style=basemap,
            tooltip=tooltip,
        )

    # Heatmap / Hexagon expect point positions
    pts = _geom_to_points(gdf)
    df = pd.concat([pts, gdf.drop(columns=['geometry'], errors='ignore')], axis=1)
    if weight_col and (weight_col in df.columns) and np.issubdtype(df[weight_col].dtype, np.number):
        w = weight_col
    else:
        w = None

    if mode == 'Heatmap':
        color_range = _interp_palette(palette, steps=6).tolist()
        layer = pdk.Layer(
            "HeatmapLayer",
            data=df,
            get_position=["lon", "lat"],
            get_weight=w if w else 1,
            radiusPixels=int(heat_radius_px),
            intensity=float(heat_intensity_val),
            colorRange=color_range,
            pickable=False,
        )
        return pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            map_provider="carto",
            map_style=basemap,
            tooltip=False,
        )

    if mode == 'Hexagon':
        kwargs = dict(
            data=df,
            get_position=["lon", "lat"],
            radius=int(hex_radius_m),
            extruded=bool(hex_extruded_val),
            elevation_scale=int(hex_elev_scale_val),
            pickable=True,
            colorRange=_interp_palette(palette, steps=6).tolist()
        )
        if w:
            kwargs["get_color_weight"] = w
            kwargs["color_aggregation"] = "SUM"
            kwargs["get_elevation_weight"] = w
            kwargs["elevation_aggregation"] = "SUM"
        layer = pdk.Layer("HexagonLayer", **kwargs)
        return pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            map_provider="carto",
            map_style=basemap,
            tooltip=True,
        )
    return None


def _css_gradient(colors: List[Tuple[int,int,int]]) -> str:
    n = len(colors)
    stops = ", ".join([f"rgb({r},{g},{b}) {int(i/(n-1)*100)}%" for i,(r,g,b) in enumerate(colors)])
    return f"background: linear-gradient(90deg, {stops}); height: 14px; border-radius: 2px;"


def _html_legend_title(title: str) -> str:
    return f"<div style='font-weight:600;margin-bottom:6px'>{title}</div>"


def _update_legend():
    """Build legend HTML based on current mode and selections (existing map tab)."""
    try:
        layer = map_layer_select.value
        if not (_current_gpkg_path and layer):
            legend_pane.object = ""
            return
        gdf = gpd.read_file(str(_current_gpkg_path), layer=layer)
        mode = map_mode.value
        pal_name = palette_select.value
        if mode == 'GeoJSON':
            # Categorical?
            if color_by.value and (color_mode.value == 'categorical' or
               (color_mode.value == 'auto' and not np.issubdtype(gdf[color_by.value].dtype, np.number))):
                s = gdf[color_by.value].astype(str)
                cats = pd.Categorical(s)
                palette = _PALETTES['Category10']
                html = _html_legend_title(f"Categorical: {color_by.value}")
                uniq = list(pd.unique(cats.codes))
                labels = list(pd.unique(cats.categories))
                items = []
                max_show = min(10, len(labels))
                for i in range(max_show):
                    rgb = palette[i % len(palette)]
                    lbl = labels[i]
                    items.append(
                        f"<div style='display:flex;align-items:center;margin:2px 0'>"
                        f"<div style='width:12px;height:12px;background:rgb({rgb[0]},{rgb[1]},{rgb[2]});"
                        f"border:1px solid #999;margin-right:6px'></div>"
                        f"<div>{lbl}</div></div>"
                    )
                if len(labels) > max_show:
                    items.append("<div style='font-size:11px;color:#666'>… more categories</div>")
                legend_pane.object = html + "".join(items)
                return
            # Continuous legend
            if color_by.value:
                s = pd.to_numeric(gdf[color_by.value], errors='coerce')
                vmin, vmax = _get_series_stats(s)
            else:
                vmin, vmax = None, None
            ramp = _interp_palette(pal_name, 16).tolist()
            html = _html_legend_title(f"Continuous: {color_by.value or 'constant'}")
            html += f"<div style='{_css_gradient(ramp)}'></div>"
            if vmin is not None and vmax is not None:
                html += ("<div style='display:flex;justify-content:space-between;font-size:12px;'>"
                         f"<span>{np.round(vmin,3)}</span><span>{np.round(vmax,3)}</span></div>")
            legend_pane.object = html
            return
        if mode == 'Heatmap':
            label = weight_by.value if weight_by.value else "count"
            ramp = _interp_palette(pal_name, 16).tolist()
            html = _html_legend_title(f"Heatmap weight: {label}")
            html += f"<div style='{_css_gradient(ramp)}'></div>"
            legend_pane.object = html
            return
        if mode == 'Hexagon':
            label = weight_by.value if weight_by.value else "count"
            ramp = _interp_palette(pal_name, 16).tolist()
            html = _html_legend_title(f"Hexagon aggregation: {label}")
            html += f"<div style='{_css_gradient(ramp)}'></div>"
            html += "<div style='font-size:12px;color:#666;margin-top:4px'>Height reflects aggregated value</div>"
            legend_pane.object = html
            return
        legend_pane.object = ""
    except Exception:
        legend_pane.object = ""

# ======================================
# NEW: Compare tab helpers
# ======================================

def _refresh_overlay_widgets():
    # Populate file and layer options for compare tab
    if not _all_files:
        over_base_file.options = []
        over_base_file.disabled = True
        over_base_layer.options = []
        over_base_layer.disabled = True
        over_layers.options = []
        over_layers.disabled = True
        return
    files = list(_all_files.keys())
    over_base_file.options = files
    over_base_file.disabled = False
    # set base file if not set
    if not over_base_file.value and files:
        over_base_file.value = files[0]

    # base layer options
    base_file = over_base_file.value
    base_layers = _layers_by_file.get(base_file, [])
    over_base_layer.options = base_layers
    over_base_layer.disabled = not bool(base_layers)
    if base_layers and (over_base_layer.value not in base_layers):
        over_base_layer.value = base_layers[0]

    # overlay list (exclude the chosen base layer)
    opts = []
    for fk, layers in _layers_by_file.items():
        for ly in layers:
            key = f"{fk}::{ly}"
            if not (fk == base_file and ly == over_base_layer.value):
                opts.append(key)
    over_layers.options = opts
    over_layers.disabled = not bool(opts)


def _compare_build_map(base_gdf: gpd.GeoDataFrame,
                       overlay_keys: List[str],
                       color_metric: Optional[str]) -> Optional[pdk.Deck]:
    """Build a multi-layer Deck with base polygons and overlay layers on top."""
    if base_gdf is None or base_gdf.empty:
        return None
    layers = []
    base_wgs = _to_wgs84(base_gdf.copy())

    # Color base by a metric if provided
    gdf = base_wgs.copy()
    if color_metric and (color_metric in gdf.columns):
        gdf["__color__"] = _colorize_series(gdf[color_metric], 'continuous', compare_palette.value, compare_opacity.value)
    else:
        base_rgb = _PALETTES['Category10'][7]
        gdf["__color__"] = [[base_rgb[0], base_rgb[1], base_rgb[2], int(255*compare_opacity.value)]] * len(gdf)

    base_layer = pdk.Layer(
        "GeoJsonLayer",
        data=gdf,
        pickable=True,
        stroked=True,
        filled=True,
        get_fill_color="__color__",
        get_line_color=[50,50,50,180],
        get_line_width=1,
    )
    layers.append(base_layer)

    # overlay layers styling
    cat = _PALETTES['Category10']
    for i, key in enumerate(overlay_keys):
        try:
            fk, ly = key.split("::", 1)
            ogdf = _read_layer(fk, ly)
            has_geom, gtype = _get_geometry_info(ogdf)
            if not has_geom:
                continue
            ogdf_wgs = _to_wgs84(ogdf)
            rgb = cat[i % len(cat)]
            alpha = 180
            if gtype in ("Point", "MultiPoint"):
                layer = pdk.Layer(
                    "GeoJsonLayer",
                    data=ogdf_wgs,
                    pickable=True,
                    filled=True,
                    stroked=False,
                    get_fill_color=[rgb[0], rgb[1], rgb[2], alpha],
                    get_point_radius=4,
                    point_radius_units="pixels",
                )
            elif gtype in ("LineString", "MultiLineString"):
                layer = pdk.Layer(
                    "GeoJsonLayer",
                    data=ogdf_wgs,
                    pickable=True,
                    filled=False,
                    stroked=True,
                    get_line_color=[rgb[0], rgb[1], rgb[2], alpha],
                    get_line_width=2,
                    line_width_units="pixels",
                )
            else:  # polygons
                layer = pdk.Layer(
                    "GeoJsonLayer",
                    data=ogdf_wgs,
                    pickable=True,
                    filled=False,
                    stroked=True,
                    get_line_color=[rgb[0], rgb[1], rgb[2], alpha],
                    get_line_width=2,
                )
            layers.append(layer)
        except Exception:
            continue
    view_state = _compute_view_state(base_wgs)
    return pdk.Deck(layers=layers, initial_view_state=view_state, map_provider="carto", map_style=basemap_style.value)


def _compare_legend_html(color_metric: Optional[str]) -> str:
    html = _html_legend_title("Compare/Overlay Legend")
    if color_metric:
        ramp = _interp_palette(compare_palette.value, 16).tolist()
        html += f"<div style='{_css_gradient(ramp)}'></div>"
        html += f"<div style='font-size:12px;margin-top:4px'>Base polygons colored by <b>{color_metric}</b></div>"
    # Overlay color swatches
    if over_layers.value:
        html += "<div style='margin-top:6px;font-weight:600'>Overlays</div>"
        cat = _PALETTES['Category10']
        for i, key in enumerate(list(over_layers.value)[:10]):
            r,g,b = cat[i % len(cat)]
            html += (f"<div style='display:flex;align-items:center;margin:2px 0'>"
                     f"<div style='width:12px;height:12px;background:rgb({r},{g},{b});border:1px solid #999;margin-right:6px'></div>"
                     f"<div>{key}</div></div>")
        if len(over_layers.value) > 10:
            html += "<div style='font-size:11px;color:#666'>… more overlays</div>"
    return html


def _geom_is_polygonal(gdf: gpd.GeoDataFrame) -> bool:
    _, gt = _get_geometry_info(gdf)
    return gt in ("Polygon", "MultiPolygon")


def _counts_points_in_polygons(base: gpd.GeoDataFrame, overlay_pts: gpd.GeoDataFrame, base_id: str) -> pd.Series:
    pts = overlay_pts.copy()
    if pts.crs is None:
        pts = pts.set_crs(4326, allow_override=True)
    pts = pts.to_crs(base.crs)
    joined = gpd.sjoin(pts[["geometry"]], base[[base_id, "geometry"]], how='left', predicate='within')
    counts = joined.groupby(base_id, dropna=False).size().rename("count")
    return counts


def _area_intersection(base: gpd.GeoDataFrame, overlay_poly: gpd.GeoDataFrame, base_id: str) -> pd.Series:
    # Project to Web Mercator for area (m^2). For better accuracy, users can preproject.
    base_wm = base.to_crs(3857)
    over_wm = overlay_poly.to_crs(3857)
    try:
        inter = gpd.overlay(base_wm[[base_id, 'geometry']], over_wm[['geometry']], how='intersection')
    except Exception:
        # Fallback via spatial index join then intersection
        idx = gpd.sjoin(base_wm[[base_id, 'geometry']], over_wm[['geometry']], how='inner', predicate='intersects')
        inter = gpd.GeoDataFrame(idx.merge(base_wm[[base_id]].reset_index(drop=True), left_index=True, right_index=True),
                                 geometry=gpd.overlay(base_wm[[base_id, 'geometry']], over_wm[['geometry']], how='intersection').geometry, crs=base_wm.crs)
    inter['__area__'] = inter.geometry.area
    s = inter.groupby(base_id)['__area__'].sum().rename('area_m2')
    return s


def _compute_overlay_metrics(event=None):
    global _compare_base_gdf
    compare_table.value = pd.DataFrame()
    compare_deck.object = None
    compare_legend.object = ""

    if not (over_base_file.value and over_base_layer.value):
        overlay_status.object = "Pick a base file and polygon layer."
        overlay_status.alert_type = "warning"
        return
    base_key = over_base_file.value
    base_layer = over_base_layer.value
    base_gdf = _read_layer(base_key, base_layer).copy()
    if base_gdf.empty:
        overlay_status.object = "Base layer is empty."
        overlay_status.alert_type = "danger"
        return
    if not _geom_is_polygonal(base_gdf):
        overlay_status.object = "Base layer must be polygonal for comparison metrics."
        overlay_status.alert_type = "danger"
        return

    # Ensure a stable id column for grouping
    if '__base_id__' not in base_gdf.columns:
        base_gdf['__base_id__'] = np.arange(len(base_gdf))
    base_id = '__base_id__'

    # Build a working result frame
    res = pd.DataFrame({base_id: base_gdf[base_id].values})
    res.set_index(base_id, inplace=True)

    # Iterate overlays
    selected = list(over_layers.value)
    if not selected:
        overlay_status.object = "Select at least one overlay layer (from any uploaded file)."
        overlay_status.alert_type = "warning"
        return

    base_proj = base_gdf
    if base_proj.crs is None:
        base_proj = base_proj.set_crs(4326, allow_override=True)

    for key in selected:
        try:
            fk, ly = key.split('::', 1)
            ogdf = _read_layer(fk, ly)
            has_geom, gtype = _get_geometry_info(ogdf)
            if not has_geom:
                continue
            ogdf = ogdf[["geometry"]].dropna().copy()
            if ogdf.empty:
                continue
            # Ensure CRS
            if ogdf.crs is None:
                ogdf = ogdf.set_crs(4326, allow_override=True)
            if _geom_is_polygonal(ogdf):
                if over_metric.value.startswith('area_intersection'):
                    s = _area_intersection(base_proj, ogdf, base_id)
                    col = f"{fk}::{ly}__area_m2"
                    res[col] = s
                else:
                    # counts not meaningful for polygon overlay; count intersections instead
                    try:
                        inter = gpd.overlay(base_proj[[base_id, 'geometry']], ogdf[['geometry']], how='intersection')
                        s = inter.groupby(base_id).size().rename('intersections')
                    except Exception:
                        s = pd.Series(dtype='int64')
                    col = f"{fk}::{ly}__intersections"
                    res[col] = s
            else:
                # treat as points/lines -> use points-in-polygons count; for lines, count segments touching
                if gtype in ("Point", "MultiPoint"):
                    s = _counts_points_in_polygons(base_proj, ogdf, base_id)
                    col = f"{fk}::{ly}__count"
                    res[col] = s
                else:
                    # lines: approximate by counting line pieces intersecting polygons
                    ogdf2 = ogdf.to_crs(base_proj.crs)
                    joined = gpd.sjoin(ogdf2, base_proj[[base_id, 'geometry']], how='left', predicate='intersects')
                    s = joined.groupby(base_id, dropna=False).size().rename('lines_intersect')
                    col = f"{fk}::{ly}__lines_intersect"
                    res[col] = s
        except Exception as e:
            # continue other overlays
            continue

    # Fill NaNs with zeros for metrics
    res = res.fillna(0)

    # Attach back to base_gdf
    out = base_gdf.merge(res.reset_index(), on=base_id, how='left')
    _compare_base_gdf = out

    # Update color-by options to any computed metric columns
    metric_cols = [c for c in out.columns if c.startswith(tuple(f"{k}__" for k in [s.split('::',1)[0] + '::' + s.split('::',1)[1] for s in selected]))]
    # Fallback: collect by suffix patterns
    if not metric_cols:
        metric_cols = [c for c in out.columns if c.endswith(('_count', '_area_m2', '_lines_intersect', '__intersections'))]

    compare_color_by.options = metric_cols
    compare_color_by.disabled = not bool(metric_cols)
    if metric_cols:
        compare_color_by.value = metric_cols[0]

    # Update table view (non-geometry preview)
    geom_col = getattr(out, 'geometry', None)
    cols_no_geom = [c for c in out.columns if c != (geom_col.name if hasattr(geom_col, 'name') else 'geometry')]
    preview = pd.DataFrame(out[cols_no_geom].head(25))
    compare_table.value = preview

    # Build map
    deck = _compare_build_map(out, selected, compare_color_by.value if metric_cols else None)
    compare_deck.object = deck
    compare_legend.object = _compare_legend_html(compare_color_by.value if metric_cols else None)

    overlay_status.object = "Overlay metrics computed."
    overlay_status.alert_type = "success"


# ======================================
# Event handlers
# ======================================

def _on_file_change(event):
    global _current_gpkg_path, _gdf_cache
    _gdf_cache.clear()
    deck_pane.object = None
    legend_pane.object = ""
    status_alert.object = "Processing upload..."
    status_alert.alert_type = "info"

    paths = _save_uploaded_files(event.new)
    if not paths:
        status_alert.object = "No file uploaded. Please drop .gpkg file(s)."
        status_alert.alert_type = "warning"
        return

    # Register files
    _all_files.clear()
    _layers_by_file.clear()
    for p in paths:
        key = p.name
        _all_files[key] = p
        try:
            _layers_by_file[key] = _list_layers(p)
        except Exception:
            _layers_by_file[key] = []

    # Keep existing single-file experience using the first file
    _current_gpkg_path = paths[0]

    # Configure existing widgets based on first file
    try:
        layers = _layers_by_file[_current_gpkg_path.name]
    except Exception:
        layers = []

    if not layers:
        for w in (layer_select, map_layer_select):
            w.options = []
            w.disabled = True
        status_alert.object = "No layers found in the first GeoPackage."
        status_alert.alert_type = "warning"
    else:
        # Scatter layer default
        layer_select.options = layers
        layer_select.value = layers[0]
        layer_select.disabled = False
        # Map layer default (independent unless sync)
        map_layer_select.options = layers
        map_layer_select.value = layers[0]
        map_layer_select.disabled = False
        # Initialize widgets from the selected layers
        try:
            gdf_scatter = gpd.read_file(str(_current_gpkg_path), layer=layer_select.value)
            _refresh_scatter_widgets(gdf_scatter)
            gdf_map = gpd.read_file(str(_current_gpkg_path), layer=map_layer_select.value)
            _refresh_map_widgets(gdf_map)
            _update_map()
            _update_legend()
        except Exception:
            pass
        status_alert.object = f"Loaded {len(paths)} file(s). Choose layers and styling, then explore!"
        status_alert.alert_type = "success"

    # Populate the compare/overlay widgets
    _refresh_overlay_widgets()


def _refresh_scatter_widgets(gdf: gpd.GeoDataFrame):
    geom_col = getattr(gdf, "geometry", None)
    cols_no_geom = [c for c in gdf.columns if c != (geom_col.name if hasattr(geom_col, 'name') else 'geometry')]
    num_cols = pd.DataFrame(gdf[cols_no_geom]).select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        x_select.options = num_cols
        y_select.options = num_cols
        x_select.value = num_cols[0]
        y_select.value = num_cols[1] if len(num_cols) > 1 else num_cols[0]
        x_select.disabled = False
        y_select.disabled = False
    else:
        x_select.options = []
        y_select.options = []
        x_select.disabled = True
        y_select.disabled = True


def _refresh_map_widgets(gdf: gpd.GeoDataFrame):
    geom_col = getattr(gdf, "geometry", None)
    cols_no_geom = [c for c in gdf.columns if c != (geom_col.name if hasattr(geom_col, 'name') else 'geometry')]
    num_cols = pd.DataFrame(gdf[cols_no_geom]).select_dtypes(include=[np.number]).columns.tolist()
    color_by.options = cols_no_geom
    color_by.value = cols_no_geom[0] if cols_no_geom else None
    color_by.disabled = not bool(cols_no_geom)
    size_by.options = [None] + num_cols
    size_by.value = None
    size_by.disabled = not bool(num_cols)
    weight_by.options = [None] + num_cols
    weight_by.value = None
    weight_by.disabled = not bool(num_cols)


def _on_scatter_layer_change(event):
    layer = event.new
    if not (_current_gpkg_path and layer):
        return
    try:
        gdf = gpd.read_file(str(_current_gpkg_path), layer=layer)
        _refresh_scatter_widgets(gdf)
        if sync_map_layer.value:
            map_layer_select.value = layer  # triggers map refresh via watch
    except Exception as e:
        x_select.options = []
        y_select.options = []
        x_select.disabled = True
        y_select.disabled = True
        status_alert.object = f"Failed to read layer '{layer}': {e}"
        status_alert.alert_type = "danger"


def _on_map_layer_change(event):
    layer = event.new
    deck_pane.object = None
    legend_pane.object = ""
    if not (_current_gpkg_path and layer):
        return
    try:
        gdf = gpd.read_file(str(_current_gpkg_path), layer=layer)
        _refresh_map_widgets(gdf)
        _update_map()
        _update_legend()
    except Exception as e:
        for w in (color_by, size_by, weight_by):
            w.options = []
            w.disabled = True
        deck_pane.object = None
        status_alert.object = f"Failed to read map layer '{layer}': {e}"
        status_alert.alert_type = "danger"


def _on_sync_toggle(event):
    if sync_map_layer.value:
        # lock map layer to scatter layer
        map_layer_select.disabled = True
        if layer_select.value != map_layer_select.value:
            map_layer_select.value = layer_select.value  # triggers update
        else:
            _on_map_layer_change(None)
    else:
        map_layer_select.disabled = False


def _update_map(*events):
    layer = map_layer_select.value
    if not (_current_gpkg_path and layer):
        deck_pane.object = None
        return
    gdf = gpd.read_file(str(_current_gpkg_path), layer=layer)
    has_geom, _ = _get_geometry_info(gdf)
    if not has_geom:
        deck_pane.object = None
        return
    deck = _build_deck(
        gdf_in=gdf,
        mode=map_mode.value,
        color_col=color_by.value,
        color_mode_val=color_mode.value,
        palette=palette_select.value,
        opacity=opacity_slider.value,
        size_col=size_by.value,
        size_scale_px=size_scale.value,
        basemap=basemap_style.value,
        outlines=show_outlines.value,
        weight_col=weight_by.value,
        heat_radius_px=heat_radius.value,
        heat_intensity_val=heat_intensity.value,
        hex_radius_m=hex_radius.value,
        hex_extruded_val=hex_extruded.value,
        hex_elev_scale_val=hex_elev_scale.value
    )
    deck_pane.object = deck


# Wire events
file_dropper.param.watch(_on_file_change, "value")
layer_select.param.watch(_on_scatter_layer_change, "value")
map_layer_select.param.watch(_on_map_layer_change, "value")
sync_map_layer.param.watch(_on_sync_toggle, "value")
for w in (color_by, color_mode, palette_select, opacity_slider,
          size_by, size_scale, basemap_style, show_outlines,
          map_mode, weight_by, heat_radius, heat_intensity,
          hex_radius, hex_extruded, hex_elev_scale):
    w.param.watch(lambda e: (_update_map(), _update_legend()), 'value')

# Compare/Overlay events
over_base_file.param.watch(lambda e: _refresh_overlay_widgets(), 'value')
over_base_layer.param.watch(lambda e: _refresh_overlay_widgets(), 'value')
over_layers.param.watch(lambda e: None, 'value')  # no-op, used when computing
compare_palette.param.watch(lambda e: _compute_overlay_metrics(), 'value')
compare_color_by.param.watch(lambda e: _compute_overlay_metrics(), 'value')
compute_btn.on_click(_compute_overlay_metrics)

# ======================================
# Scatter plot view
# ======================================
@pn.depends(layer_select, x_select, y_select)
def view_plot(layer, x_col, y_col):
    if not _current_gpkg_path:
        return pn.pane.Markdown("Upload a GeoPackage to begin.", sizing_mode="stretch_width")
    if not (layer and x_col and y_col):
        return pn.pane.Markdown("Pick a table, then select X and Y numeric columns.", sizing_mode="stretch_width")
    try:
        gdf = gpd.read_file(str(_current_gpkg_path), layer=layer)
        geom_col = getattr(gdf, "geometry", None)
        cols = [c for c in gdf.columns if c != (geom_col.name if hasattr(geom_col, 'name') else 'geometry')]
        df = pd.DataFrame(gdf[cols])
        if x_col not in df.columns or y_col not in df.columns:
            return pn.pane.Alert("Selected columns not found. Pick X/Y again.", alert_type="warning")
        df2 = df[[x_col, y_col]].apply(pd.to_numeric, errors='coerce').dropna()
        scatter = hv.Scatter(df2, kdims=[x_col], vdims=[y_col])
        return scatter.opts(
            height=500, responsive=True, size=6, alpha=0.6, color="#1f77b4",
            tools=["hover", "box_zoom", "wheel_zoom", "pan", "reset"],
        )
    except Exception as e:
        return pn.pane.Alert(f"Plot error: {e}", alert_type="danger", sizing_mode="stretch_width")


@pn.depends(layer_select)
def layer_preview(layer):
    if not (_current_gpkg_path and layer):
        return pn.Spacer(height=0)
    gdf = gpd.read_file(str(_current_gpkg_path), layer=layer)
    geom_col = getattr(gdf, "geometry", None)
    cols = [c for c in gdf.columns if c != (geom_col.name if hasattr(geom_col, "name") else 'geometry')]
    preview = pd.DataFrame(gdf[cols].head(10))
    return pn.widgets.Tabulator(preview, disabled=True, height=220, sizing_mode="stretch_width")

# ======================================
# Layout / Template
# ======================================
scatter_controls = pn.Row(layer_select, x_select, y_select, sizing_mode="stretch_width")

map_controls_left = pn.Column(
    pn.pane.Markdown("### Map Controls", margin=(0,0,5,0)),
    sync_map_layer,
    map_layer_select,
    map_mode,
    pn.Spacer(height=5),
    pn.pane.Markdown("**Symbology**"),
    color_by, color_mode, palette_select, opacity_slider,
    size_by, size_scale,
    pn.pane.Markdown("**Basemap & Style**"),
    basemap_style, show_outlines,
    pn.Spacer(height=10),
    pn.pane.Markdown("**Heatmap**"),
    weight_by, heat_radius, heat_intensity,
    pn.pane.Markdown("**Hexagon**"),
    hex_radius, hex_extruded, hex_elev_scale,
    pn.pane.Markdown("**Legend**"),
    legend_pane,
    width=340,
)
map_controls = pn.Row(map_controls_left, deck_pane, sizing_mode="stretch_both")

# NEW: Compare/Overlay tab layout
compare_controls_left = pn.Column(
    pn.pane.Markdown("### Compare / Overlays (multi-GPKG)"),
    overlay_status,
    pn.pane.Markdown("**Base (polygons)**"),
    over_base_file, over_base_layer,
    pn.pane.Markdown("**Overlays**"),
    over_layers,
    pn.pane.Markdown("**Metric & Styling**"),
    over_metric,
    compare_palette, compare_opacity,
    compute_btn,
    pn.pane.Markdown("**Legend**"),
    compare_legend,
    width=360
)
compare_area = pn.Row(compare_controls_left, compare_deck, sizing_mode="stretch_both")
compare_bottom = pn.Column(pn.pane.Markdown("#### Preview (first 25 rows, no geometry)"), compare_table)
compare_tab = pn.Column(compare_area, pn.Spacer(height=10), compare_bottom, sizing_mode="stretch_both")


tabs = pn.Tabs(
    ("Scatter (HoloViews)", pn.Column(scatter_controls, view_plot, sizing_mode="stretch_both")),
    ("Map (pydeck)", map_controls),
    ("Compare/Overlays", compare_tab),
    dynamic=True,
    sizing_mode="stretch_both"
)

main = pn.Column(
    file_dropper,
    status_alert,
    tabs,
    pn.Spacer(height=10),
    layer_preview,
    sizing_mode="stretch_both",
)

template = pn.template.FastListTemplate(
    title="GeoPackage Explorer: Scatter + Map + Compare",
    sidebar=[],
    main=[main],
)

template.servable()
