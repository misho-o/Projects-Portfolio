import streamlit as st
import networkx as nx
from collections import deque
import heapq
import math

# ─────────────────────────────────────────────
# 1. CAMPUS DATA
# ─────────────────────────────────────────────
locations = {
    "Main Building":     {"pos": (200, 150)},
    "Science Lab":       {"pos": (370, 100)},
    "Accounts Office":   {"pos": (480, 220)},
    "Football Ground 1": {"pos": (360, 310)},
    "Football Ground 2": {"pos": (220, 300)},
}

G = nx.Graph()
edges = [
    ("Main Building",     "Science Lab",       50),
    ("Science Lab",       "Accounts Office",   70),
    ("Accounts Office",   "Football Ground 1", 40),
    ("Football Ground 1", "Football Ground 2", 30),
    ("Main Building",     "Football Ground 2", 100),
    ("Main Building",     "Football Ground 1", 120),
]
for u, v, w in edges:
    G.add_edge(u, v, weight=w)

# ─────────────────────────────────────────────
# 2. ALGORITHM IMPLEMENTATIONS
# ─────────────────────────────────────────────

def heuristic(a, b):
    """Euclidean distance between two nodes using pixel positions as a heuristic."""
    x1, y1 = locations[a]["pos"]
    x2, y2 = locations[b]["pos"]
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def astar_path(graph, src, dst):
    """
    A* Search: f(n) = g(n) + h(n)
    g(n) = actual cost so far (edge weights)
    h(n) = heuristic estimate to goal (Euclidean distance)
    Optimal and faster than BFS/DFS because the heuristic guides the search.
    """
    open_set = [(0 + heuristic(src, dst), 0, src, [src])]
    visited = set()
    while open_set:
        f, g, node, path = heapq.heappop(open_set)
        if node in visited:
            continue
        visited.add(node)
        if node == dst:
            return path
        for neighbor in graph.neighbors(node):
            if neighbor not in visited:
                new_g = g + graph[node][neighbor]['weight']
                new_f = new_g + heuristic(neighbor, dst)
                heapq.heappush(open_set, (new_f, new_g, neighbor, path + [neighbor]))
    return [src, dst]


def bfs_path(graph, src, dst):
    """
    BFS: Explores all neighbors at current depth before going deeper.
    Finds fewest hops (ignores weights). Uses a Queue (FIFO).
    """
    visited = {src}
    queue = deque([[src]])
    while queue:
        path = queue.popleft()
        node = path[-1]
        if node == dst:
            return path
        for neighbor in graph.neighbors(node):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(path + [neighbor])
    return [src, dst]


def dfs_path(graph, src, dst):
    """
    DFS: Goes as deep as possible before backtracking.
    Uses a Stack (LIFO). Not optimal — purely educational.
    """
    stack = [[src]]
    visited = {src}
    while stack:
        path = stack.pop()
        node = path[-1]
        if node == dst:
            return path
        for neighbor in graph.neighbors(node):
            if neighbor not in visited:
                visited.add(neighbor)
                stack.append(path + [neighbor])
    return [src, dst]


def path_cost(graph, path):
    """Calculate total weight of a path."""
    return sum(graph[path[i]][path[i+1]]['weight'] for i in range(len(path) - 1))


def compute_route(graph, start, destinations, path_fn):
    """
    Greedy nearest-neighbor multi-stop routing.
    At each step, visit the closest unvisited destination using the chosen algorithm.
    """
    current = start
    unvisited = list(destinations)
    route = [start]
    total = 0

    while unvisited:
        next_node = min(unvisited, key=lambda t: path_cost(graph, path_fn(graph, current, t)))
        path = path_fn(graph, current, next_node)
        total += path_cost(graph, path)
        route.extend(path[1:])
        current = next_node
        unvisited.remove(next_node)

    return route, total


ALGORITHMS = {
    "A* (Smart Shortest Path)": {
        "fn": astar_path,
        "emoji": "⭐",
        "desc": "Uses actual cost + a smart estimate (heuristic) to guide the search. Fastest and most optimal.",
        "detail": "f(n) = g(n) + h(n) — balances known cost with estimated remaining cost.",
        "color": "#FFD700",
    },
    "BFS (Fewest Hops)": {
        "fn": bfs_path,
        "emoji": "🌊",
        "desc": "Explores level by level. Finds the path with fewest stops, ignoring distances.",
        "detail": "Uses a Queue (FIFO). Guaranteed to find the fewest-hop path.",
        "color": "#4AFF91",
    },
    "DFS (Deep Exploration)": {
        "fn": dfs_path,
        "emoji": "🔦",
        "desc": "Dives deep into one branch before backtracking. Not optimal — purely educational.",
        "detail": "Uses a Stack (LIFO). Order depends on adjacency list structure.",
        "color": "#FF7755",
    },
}

# ─────────────────────────────────────────────
# 3. SVG MAP
# ─────────────────────────────────────────────

def build_svg_map(route, start, destinations, algo_color="#4A9EFF"):
    W, H = 640, 400
    svg = [
        f'<svg width="{W}" height="{H}" xmlns="http://www.w3.org/2000/svg" '
        f'style="background:#0d1117; border-radius:14px; display:block;">'
    ]

    # Subtle grid
    svg.append('<g opacity="0.05">')
    for x in range(0, W, 40):
        svg.append(f'<line x1="{x}" y1="0" x2="{x}" y2="{H}" stroke="#fff" stroke-width="1"/>')
    for y in range(0, H, 40):
        svg.append(f'<line x1="0" y1="{y}" x2="{W}" y2="{y}" stroke="#fff" stroke-width="1"/>')
    svg.append('</g>')

    # All edges (dim dashed)
    for u, v, d in G.edges(data=True):
        x1, y1 = locations[u]["pos"]
        x2, y2 = locations[v]["pos"]
        svg.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            f'stroke="#ffffff18" stroke-width="2" stroke-dasharray="5,4"/>'
        )
        mx, my = (x1 + x2) // 2, (y1 + y2) // 2
        svg.append(
            f'<text x="{mx}" y="{my - 5}" fill="#ffffff33" font-size="10" '
            f'text-anchor="middle" font-family="monospace">{d["weight"]}m</text>'
        )

    # Route edges (bright)
    for i in range(len(route) - 1):
        x1, y1 = locations[route[i]]["pos"]
        x2, y2 = locations[route[i + 1]]["pos"]
        svg.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            f'stroke="{algo_color}" stroke-width="5" stroke-linecap="round" opacity="0.9"/>'
        )
        mx, my = (x1 + x2) // 2, (y1 + y2) // 2
        svg.append(f'<circle cx="{mx}" cy="{my}" r="4" fill="{algo_color}" opacity="0.7"/>')

    # Nodes
    for name, info in locations.items():
        x, y = info["pos"]
        if name == start:
            fill, size = "#00FF88", 18
        elif name in destinations:
            fill, size = "#FF5555", 16
        else:
            fill, size = "#4a5568", 12

        if name in route:
            svg.append(f'<circle cx="{x}" cy="{y}" r="{size + 8}" fill="{fill}" opacity="0.15"/>')

        svg.append(
            f'<circle cx="{x}" cy="{y}" r="{size}" fill="{fill}" stroke="#0d1117" stroke-width="3"/>'
        )

        if name in route:
            step_num = route.index(name) + 1
            svg.append(
                f'<text x="{x}" y="{y + 4}" fill="#0d1117" font-size="11" '
                f'font-weight="bold" text-anchor="middle" font-family="monospace">{step_num}</text>'
            )

        label = name.replace("Football Ground", "Ground")
        svg.append(
            f'<text x="{x}" y="{y + size + 16}" fill="#ffffffbb" font-size="11" '
            f'text-anchor="middle" font-family="monospace">{label}</text>'
        )

    # Legend
    legend = [("● Start", "#00FF88"), ("● Destination", "#FF5555"),
               ("● Other node", "#4a5568"), ("─ Route", algo_color)]
    for i, (lbl, col) in enumerate(legend):
        svg.append(
            f'<text x="12" y="{H - 60 + i * 16}" fill="{col}" '
            f'font-size="11" font-family="monospace">{lbl}</text>'
        )

    svg.append('</svg>')
    return "\n".join(svg)

# ─────────────────────────────────────────────
# 4. STREAMLIT UI
# ─────────────────────────────────────────────

st.set_page_config(page_title="Campus Navigator", layout="wide", page_icon="🏫")

st.markdown("""
<style>
    .stApp { background-color: #0d1117; }
    .block-container { padding-top: 1.5rem; }
    section[data-testid="stSidebar"] {
        background: #161b22;
        border-right: 1px solid #30363d;
    }
    .stButton > button {
        background: #238636 !important;
        color: white !important;
        border: none !important;
        font-family: monospace !important;
        font-weight: bold !important;
        border-radius: 6px !important;
    }
    .stButton > button:hover { background: #2ea043 !important; }
</style>
""", unsafe_allow_html=True)

st.title("🏫 Campus Route Optimizer")
st.caption("Check your destinations, enter tasks, pick an algorithm, find your route.")

# ── Sidebar ──
st.sidebar.header("⚙️ Settings")

all_buildings = list(locations.keys())
start_point = st.sidebar.selectbox("📍 You are currently at:", options=all_buildings)

st.sidebar.markdown("---")
st.sidebar.markdown("**🎯 Pick destinations & enter your task there:**")

selected_tasks = []
task_notes = {}

for building in all_buildings:
    if building == start_point:
        continue
    chk_col, txt_col = st.sidebar.columns([1, 3])
    with chk_col:
        # Small vertical space so checkbox aligns with text input
        st.write("")
        chosen = st.checkbox(label=building, key=f"chk_{building}", label_visibility="collapsed")
    with txt_col:
        task_input = st.text_input(
            label=building,
            placeholder=building,
            key=f"task_{building}",
            label_visibility="collapsed"
        )
    if chosen:
        selected_tasks.append(building)
        task_notes[building] = task_input.strip() if task_input.strip() else "(no task entered)"

st.sidebar.markdown("---")
algo_name = st.sidebar.selectbox("🧠 Algorithm:", options=list(ALGORITHMS.keys()))
algo = ALGORITHMS[algo_name]

# Native Streamlit info box — no HTML rendering bugs
st.sidebar.info(f"{algo['emoji']} **{algo_name}**\n\n{algo['desc']}\n\n_{algo['detail']}_")

run = st.sidebar.button("🚀 Find Optimal Route", use_container_width=True)

# ── Main Panel ──
if run:
    if not selected_tasks:
        st.warning("Please check at least one destination in the sidebar.")
    else:
        route, total_dist = compute_route(G, start_point, selected_tasks, algo["fn"])

        col1, col2 = st.columns([1, 2], gap="large")

        with col1:
            st.subheader("📋 Journey Plan")
            st.markdown(f"**Algorithm:** `{algo_name}`")
            st.markdown(f"**Total Distance:** `{total_dist} meters`")
            st.markdown(f"**Total Stops:** `{len(route)}`")
            st.divider()

            for i, step in enumerate(route):
                if step == start_point:
                    icon, note = "🟢", "Starting point"
                elif step in selected_tasks:
                    icon, note = "🔴", task_notes.get(step, "")
                else:
                    icon, note = "🔵", "passing through"
                st.markdown(f"**{i + 1}. {icon} {step}**")
                st.caption(f"   ↳ {note}")

        with col2:
            st.subheader("🗺️ Campus Map")
            svg = build_svg_map(route, start_point, selected_tasks, algo_color=algo["color"])
            st.markdown(svg, unsafe_allow_html=True)
            st.divider()
            st.caption("💡 Tip: Switch algorithms to see different routes. A* is optimal, BFS minimises hops, DFS is unpredictable.")

else:
    st.info("👈 Use the sidebar: pick your location, check destinations, enter your task at each one, then hit **Find Optimal Route**.")
    st.divider()

    # Algorithm explainer — pure Streamlit, no HTML
    st.subheader("🧠 Available Algorithms")
    cols = st.columns(3)
    for i, (name, info) in enumerate(ALGORITHMS.items()):
        with cols[i]:
            st.markdown(f"### {info['emoji']} {name}")
            st.markdown(info["desc"])
            st.caption(info["detail"])