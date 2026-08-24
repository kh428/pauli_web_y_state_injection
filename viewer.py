"""Self-contained multi-scene 3D HTML viewers for the injection Pauli webs.

Two viewer styles:
  write_viewer       one scene at a time (radio buttons)
  write_multi_viewer a graph plus individually toggleable webs (check any combination)

The three.js viewer code (`zx_viewer_3D.js`) is by Aleks Kissinger / John van de
Wetering et al., Apache-2.0, taken from

    https://github.com/kh428/pyzx_3d_viewer

and inlined at generation time so the produced .html needs nothing but a browser
(three.js itself is pulled from a CDN on first render, so the page needs the net
once). Note: the web viewer has no purple-box rendering; these scenes are plain
spiders plus Pauli webs, which it does support.
"""
from __future__ import annotations

import json
import os

# where to find zx_viewer_3D.js; override with ZX_VIEWER_JS if you cloned the repo
JS_CANDIDATES = [
    os.environ.get('ZX_VIEWER_JS', ''),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'zx_viewer_3D.js'),
]
# The viewer JS can be inlined from a local copy (self-contained, best for an archival
# artifact) or imported straight from the repo over jsDelivr (nothing local needed).
# jsDelivr serves GitHub with the correct JavaScript MIME type; raw.githubusercontent.com
# does NOT (it sends text/plain with nosniff, so the browser refuses it as a module).
# `main` is mutable -- pin a tag or commit SHA for anything you cite.
CDN_REF = 'afee82961de136b023fb2633ff46642b9d8ee10e'  # pinned commit of kh428/pyzx_3d_viewer
CDN_URL = f'https://cdn.jsdelivr.net/gh/kh428/pyzx_3d_viewer@{CDN_REF}/zx_viewer_3D.js'

IMPORTMAP = {"imports": {
    "three": "https://cdn.jsdelivr.net/npm/three@0.172.0/build/three.module.js",
    "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.172.0/examples/jsm/",
    "d3": "https://cdn.jsdelivr.net/npm/d3@5.16.0/dist/d3.min.js"}}

_CSS = """
 body { font-family: -apple-system, Helvetica, Arial, sans-serif; margin: 18px;
        color: #1a1a1a; background: #fff; }
 h2 { margin: 0 0 6px; font-size: 19px; }
 p.sub { margin: 0 0 12px; color: #555; font-size: 13.5px; max-width: 74em; }
 button { font: inherit; padding: 5px 11px; margin: 0 5px 5px 0; cursor: pointer;
          border: 1px solid #bbb; background: #f4f4f4; border-radius: 5px; }
 button.on { background: #1b6ca8; color: #fff; border-color: #1b6ca8; }
 button.prebtn { font-size: 12.5px; padding: 4px 10px; background: #eef4fa;
                 border-color: #9dbcd6; }
 button.wbtn { font-size: 12.5px; padding: 4px 9px; font-family: ui-monospace, Menlo, monospace; }
 button.wbtn.det { border-color: #2e9e4f; }
 button.wbtn.on { background: #1b6ca8; color: #fff; border-color: #1b6ca8; }
 button.wbtn.det.on { background: #2e9e4f; border-color: #2e9e4f; }
 table { border-collapse: collapse; font-size: 13px; margin: 10px 0 14px; }
 th, td { border: 1px solid #ddd; padding: 3px 9px; text-align: left; }
 th { background: #f7f7f7; }
 .scene { border: 1px solid #e2e2e2; border-radius: 6px; min-height: 560px; }
 .grp { margin: 8px 0 4px; }
 .lbl { font-size: 12.5px; color: #666; margin: 10px 0 3px; }
 #status { font-size: 12.5px; color: #444; margin: 8px 0 4px; min-height: 1.2em; }
"""



_ATTRIB = ("\n<!-- This page embeds or imports zx_viewer_3D.js from\n"
           "     https://github.com/kh428/pyzx_3d_viewer (Ainhoa Zapirain, Kwok Ho Wan,\n"
           "     Zhenghao Zhong), licensed under the Apache License 2.0. -->\n")

def _js():
    for p in JS_CANDIDATES:
        if p and os.path.exists(p):
            with open(p) as f:
                return f.read()
    raise FileNotFoundError(
        'zx_viewer_3D.js not found. Clone https://github.com/kh428/pyzx_3d_viewer '
        'and set ZX_VIEWER_JS to its zx_viewer_3D.js.')


def _js_block(js):
    """The module prelude that makes showGraph3D available.

    js='inline' reads the local zx_viewer_3D.js and embeds it (no network for the viewer
    itself, though three.js still comes from a CDN at view time).
    js='cdn' imports it from https://github.com/kh428/pyzx_3d_viewer over jsDelivr, so
    nothing has to exist locally.
    """
    if js == 'cdn':
        return f'import {{ showGraph3D }} from "{CDN_URL}";'
    try:
        return _js()
    except FileNotFoundError:
        print(f'note: no local zx_viewer_3D.js found, falling back to the CDN ({CDN_URL}); '
              f'the page will need the network at view time. Set ZX_VIEWER_JS or place '
              f'zx_viewer_3D.js next to viewer.py for a self-contained page.')
        return f'import {{ showGraph3D }} from "{CDN_URL}";'


def graph_payload(g):
    """The {nodes, links} the viewer consumes (webs are supplied separately)."""
    def ph(v):
        p = g.phase(v)
        return '' if p == 0 else str(p)

    vs = list(g.vertices())
    minx, maxx = min(g.row(v) for v in vs), max(g.row(v) for v in vs)
    miny, maxy = min(g.qubit(v) for v in vs), max(g.qubit(v) for v in vs)
    zz = [g.vdata(v, 'z', 0.0) for v in vs]
    minz, maxz = min(zz), max(zz)
    nodes = [{'name': str(v),
              'x': float(g.row(v) - (minx + maxx) / 2),
              'y': float(g.qubit(v) - (miny + maxy) / 2),
              'z': float(g.vdata(v, 'z', 0.0) - (minz + maxz) / 2),
              't': int(g.type(v)), 'phase': ph(v), 'ground': False} for v in vs]
    links, counts = [], {}
    for e in g.edges():
        s, t = str(g.edge_s(e)), str(g.edge_t(e))
        i = counts.get((s, t), 0)
        links.append({'source': s, 'target': t, 't': int(g.edge_type(e)), 'index': i})
        counts[(s, t)] = i + 1
    return {'nodes': nodes, 'links': links, 'pauli_web': []}


def web_halfedges(web):
    """edge dict {(u,v): 'X'|'Y'|'Z'} -> the half-edge list the viewer wants."""
    out = []
    for (s, t), p in web.items():
        out.append({'source': str(s), 'target': str(t), 't': p})
        out.append({'source': str(t), 'target': str(s), 't': p})
    return out


def scene_payload(g, webs=None):
    d = graph_payload(g)
    for k, web in enumerate(webs or []):
        for he in web_halfedges(web):
            he['web'] = k
            d['pauli_web'].append(he)
    return d


# ------------------------------------------------------- one scene at a time
def write_viewer(path, title, subtitle, scenes, table_rows=None,
                 node_size=0.13, edge_radius=0.028, js='inline'):
    """scenes: list of (key, button_label, graph, [web_dicts])."""
    sc, sz, buttons, divs = {}, {}, [], []
    for key, label, g, webs in scenes:
        sc[key] = json.dumps(scene_payload(g, webs))
        sz[key] = [node_size, edge_radius]
        buttons.append(f'<button class="cbtn" data-k="{key}">{label}</button>')
        divs.append(f'<div class="scene" id="s-{key}" style="display:none"></div>')

    tbl = ''
    if table_rows:
        head = ''.join(f'<th>{h}</th>' for h in table_rows[0])
        body = ''.join('<tr>' + ''.join(f'<td>{c}</td>' for c in r) + '</tr>'
                       for r in table_rows[1:])
        tbl = f'<table><tr>{head}</tr>{body}</table>'

    payload = json.dumps(sc).replace('<', '\\u003c')
    html = f"""<!doctype html>{_ATTRIB}<html><head><meta charset="utf-8">
<title>{title}</title><style>{_CSS}</style></head><body>
<h2>{title}</h2>
<p class="sub">{subtitle}</p>
{tbl}
<div>{''.join(buttons)}</div>
{''.join(divs)}
<script type="importmap">{json.dumps(IMPORTMAP)}</script>
<script type="module">
{_js_block(js)}
window.__SC = {payload};
window.__SZ = {json.dumps(sz)};
window.__built = {{}};
window.__show = function (k) {{
  document.querySelectorAll('.scene').forEach(d => d.style.display = 'none');
  const host = document.getElementById('s-' + k);
  host.style.display = '';
  if (!window.__built[k]) {{
    host.innerHTML = '<div id="g-' + k + '"></div>';
    const s = window.__SZ[k];
    showGraph3D('g-' + k, JSON.parse(window.__SC[k]), s[0], s[1], 1.0, false, 0.0);
    window.__built[k] = true;
  }}
}};
document.querySelectorAll('.cbtn').forEach(b => b.onclick = () => {{
  document.querySelectorAll('.cbtn').forEach(x => x.classList.remove('on'));
  b.classList.add('on');
  window.__show(b.dataset.k);
}});
document.querySelector('.cbtn').click();
</script></body></html>"""
    with open(path, 'w') as f:
        f.write(html)
    return path


# ------------------------------------------- toggle any combination of webs
def write_multi_viewer(path, title, subtitle, pages, table_rows=None,
                       node_size=0.13, edge_radius=0.028, web_offset=0.7, js='inline'):
    """Pick a graph, then toggle any combination of its webs on and off.

    pages: list of (key, page_label, graph, groups, presets) where groups is a list of
           (group_label, [(web_label, web_dict, is_detector, tooltip), ...]) and presets is
           a list of (preset_label, [web indices]) offered as one-click selections.
    Selected webs are re-indexed from 0 on every rebuild, so the viewer's nested
    tube radii stay distinct for small selections.
    """
    G, W, P, buttons, panels = {}, {}, {}, [], []
    for key, label, g, groups, presets in pages:
        P[key] = [{'label': pl, 'idx': list(ix)} for pl, ix in presets]
        G[key] = json.dumps(graph_payload(g))
        W[key] = [{'label': wl, 'det': bool(det), 'tip': tip,
                   'he': web_halfedges(web)}
                  for _, items in groups for (wl, web, det, tip) in items]
        buttons.append(f'<button class="cbtn" data-k="{key}">{label}</button>')
        # per-page toggle panel, grouped
        rows, n = [], 0
        for glabel, items in groups:
            rows.append(f'<div class="lbl">{glabel}</div><div class="grp">')
            for (wl, web, det, tip) in items:
                cls = 'wbtn det' if det else 'wbtn'
                rows.append(f'<button class="{cls}" data-k="{key}" data-i="{n}" '
                            f'title="{tip}">{wl}</button>')
                n += 1
            rows.append('</div>')
        pre = ''.join(f'<button class="prebtn" data-k="{key}" data-p="{pi}">{pl["label"]}</button>'
                      for pi, pl in enumerate(P[key]))
        panels.append(
            f'<div class="panel" id="p-{key}" style="display:none">'
            f'<div class="lbl">presets</div><div class="grp">{pre}'
            f'<button class="offbtn" data-k="{key}">spread webs apart: on</button></div>'
            + ''.join(rows) + '</div>')

    tbl = ''
    if table_rows:
        head = ''.join(f'<th>{h}</th>' for h in table_rows[0])
        body = ''.join('<tr>' + ''.join(f'<td>{c}</td>' for c in r) + '</tr>'
                       for r in table_rows[1:])
        tbl = f'<table><tr>{head}</tr>{body}</table>'

    gp = json.dumps(G).replace('<', '\\u003c')
    wp = json.dumps(W).replace('<', '\\u003c')
    pp = json.dumps(P).replace('<', '\\u003c')
    html = f"""<!doctype html>{_ATTRIB}<html><head><meta charset="utf-8">
<title>{title}</title><style>{_CSS}</style></head><body>
<h2>{title}</h2>
<p class="sub">{subtitle}</p>
{tbl}
<div>{''.join(buttons)}</div>
{''.join(panels)}
<div id="status"></div>
<div class="scene" id="scene"></div>
<script type="importmap">{json.dumps(IMPORTMAP)}</script>
<script type="module">
{_js_block(js)}
window.__G = {gp};
window.__W = {wp};
window.__P = {pp};
window.__page = null;
window.__sel = {{}};
window.__off = {{}};

function rebuild() {{
  const k = window.__page;
  const g = JSON.parse(window.__G[k]);
  const picked = window.__sel[k];
  g.pauli_web = [];
  picked.forEach((idx, j) => {{
    window.__W[k][idx].he.forEach(h => {{
      g.pauli_web.push({{source: h.source, target: h.target, t: h.t, web: j}});
    }});
  }});
  const host = document.getElementById('scene');
  host.innerHTML = '<div id="gcanvas"></div>';
  showGraph3D('gcanvas', g, {node_size}, {edge_radius}, 1.0, false,
              window.__off[k] ? {web_offset} : 0.0);
  const names = picked.map(i => window.__W[k][i].label);
  document.getElementById('status').textContent =
    picked.length ? (picked.length + ' web' + (picked.length > 1 ? 's' : '') +
                     ' shown: ' + names.join(', '))
                  : 'no webs selected — the bare spacetime diagram';
}}

window.__showPage = function (k) {{
  window.__page = k;
  if (!window.__sel[k]) window.__sel[k] = [];
  if (window.__off[k] === undefined) window.__off[k] = true;
  document.querySelectorAll('.panel').forEach(d => d.style.display = 'none');
  document.getElementById('p-' + k).style.display = '';
  document.querySelectorAll('.offbtn').forEach(b => {{
    if (b.dataset.k === k) b.textContent = 'spread webs apart: ' + (window.__off[k] ? 'on' : 'off');
  }});
  rebuild();
}};

document.querySelectorAll('.cbtn').forEach(b => b.onclick = () => {{
  document.querySelectorAll('.cbtn').forEach(x => x.classList.remove('on'));
  b.classList.add('on');
  window.__showPage(b.dataset.k);
}});
document.querySelectorAll('.wbtn').forEach(b => b.onclick = () => {{
  const k = b.dataset.k, i = parseInt(b.dataset.i, 10);
  const s = window.__sel[k];
  const at = s.indexOf(i);
  if (at >= 0) {{ s.splice(at, 1); b.classList.remove('on'); }}
  else {{ s.push(i); b.classList.add('on'); }}
  rebuild();
}});
document.querySelectorAll('.prebtn').forEach(b => b.onclick = () => {{
  const k = b.dataset.k;
  const pick = window.__P[k][parseInt(b.dataset.p, 10)].idx;
  window.__sel[k] = pick.slice();
  document.querySelectorAll('.wbtn').forEach(x => {{
    if (x.dataset.k === k) x.classList.toggle('on', pick.indexOf(parseInt(x.dataset.i, 10)) >= 0);
  }});
  rebuild();
}});
document.querySelectorAll('.offbtn').forEach(b => b.onclick = () => {{
  const k = b.dataset.k;
  window.__off[k] = !window.__off[k];
  b.textContent = 'spread webs apart: ' + (window.__off[k] ? 'on' : 'off');
  rebuild();
}});
document.querySelector('.cbtn').click();
</script></body></html>"""
    with open(path, 'w') as f:
        f.write(html)
    return path


# ------------------------------------------------ inline display in a notebook
_UID = [0]


def inline_view(g, webs=None, node_size=0.16, edge_radius=0.05, camera_zoom=1.0,
                labels=False, web_offset=0.0, js='cdn', height=560):
    """Render one graph (+ optional webs) inline in a notebook.

    Returns an IPython HTML object -- `display()` it, or leave it as the last
    expression of a cell. With js='cdn' (the default) nothing needs to exist
    locally: the viewer is imported from kh428/pyzx_3d_viewer over jsDelivr.

    webs: a single {(u, v): 'X'|'Y'|'Z'} dict, or a list of them.
    """
    from IPython.display import HTML

    if webs is None:
        webs = []
    elif isinstance(webs, dict):
        webs = [webs]
    _UID[0] += 1
    tag = f'zxv{_UID[0]}'
    payload = json.dumps(json.dumps(scene_payload(g, webs))).replace('<', '\\u003c')
    return HTML(f"""
<div style="overflow:auto;background:#fff;min-height:{height}px" id="{tag}"></div>
<script type="importmap">{json.dumps(IMPORTMAP)}</script>
<script type="module">
{_js_block(js)}
showGraph3D('{tag}', JSON.parse({payload}), {node_size}, {edge_radius},
            {camera_zoom}, {'true' if labels else 'false'}, {web_offset});
</script>""")
