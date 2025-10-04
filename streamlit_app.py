# streamlit_app.py
# Aegis-INTENT lightweight interactive demo (browser-based).
# Simulates: SCADA nodes (mesh) -> attack -> AI detect -> trust drop -> intent -> orchestrator -> honeypot -> reintegrate
# Designed for Streamlit Community Cloud / Replit.

import streamlit as st
import numpy as np
import pandas as pd
import networkx as nx
import time
from datetime import datetime
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", page_title="Aegis-INTENT Demo", page_icon="🛡️")

st.title("Aegis-INTENT — Interactive Demo (Browser-only, No Local Installs) 🛡️")
st.markdown("**Simulated SCADA Mesh** → Attack → AI detect → Trust drop → Intent → Orchestration → Honeypot → Reintegration")

# Sidebar config
st.sidebar.header("Demo Controls")
num_nodes = st.sidebar.slider("Number of SCADA nodes", min_value=3, max_value=8, value=5)
attack_node = st.sidebar.selectbox("Choose node to attack (initial)", [f"node{i}" for i in range(1,num_nodes+1)], index=0)
attack_strength = st.sidebar.slider("Attack intensity (how 'noisy')", 1, 10, 6)
auto_flow = st.sidebar.selectbox("Auto demo flow", ["Manual (step buttons)","Auto run (recommended)"], index=1)

st.markdown("---")

# initialize session state
if "nodes" not in st.session_state:
    nodes = [f"node{i}" for i in range(1,num_nodes+1)]
    st.session_state.nodes = nodes
    # base trust scores (1.0 = fully trusted)
    st.session_state.trust = {n: 0.95 for n in nodes}
    st.session_state.status = {n: "trusted" for n in nodes}
    st.session_state.log = []
    st.session_state.attack_active = False
    st.session_state.honeypot_active = False
    st.session_state.attacker_history = []
    st.session_state.time0 = datetime.utcnow().isoformat()

# allow resizing nodes if user changes slider
if len(st.session_state.nodes) != num_nodes:
    nodes = [f"node{i}" for i in range(1,num_nodes+1)]
    st.session_state.nodes = nodes
    # reset trust / status
    st.session_state.trust = {n: 0.95 for n in nodes}
    st.session_state.status = {n: "trusted" for n in nodes}
    st.session_state.log = []
    st.session_state.attack_active = False
    st.session_state.honeypot_active = False
    st.session_state.attacker_history = []

# helper functions
def log(msg):
    t = datetime.utcnow().strftime("%H:%M:%S")
    st.session_state.log.insert(0, f"[{t}] {msg}")

def draw_mesh(trust_map):
    G = nx.Graph()
    nodes = list(trust_map.keys())
    G.add_nodes_from(nodes)
    # make ring + random chords for visually nice mesh
    for i in range(len(nodes)):
        G.add_edge(nodes[i], nodes[(i+1) % len(nodes)])
    # add some diagonals
    for i in range(len(nodes)//2):
        G.add_edge(nodes[i], nodes[(i+2) % len(nodes)])
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(7,4))
    # node colors by trust
    colors = []
    sizes = []
    for n in nodes:
        t = trust_map[n]
        # color mapping: red-ish low, green high
        colors.append((1.0 - t, t*0.6, 0.1))
        sizes.append(400 * (0.5 + t))
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=sizes)
    nx.draw_networkx_labels(G, pos, font_size=9)
    nx.draw_networkx_edges(G, pos, alpha=0.4)
    plt.axis('off')
    st.pyplot(plt.gcf())
    plt.close()

# small "AI" model: simple anomaly function
def anomaly_score(node, intensity):
    # Anomaly grows with intensity and randomness
    base = np.clip(np.random.normal(loc=intensity*0.08, scale=0.05), 0, 1)
    return float(np.tanh(base*2.5))

def propagate_trust_drop(target, drop_amount=0.45):
    # target drops a lot; neighbors drop proportionally (simulate GNN)
    st.session_state.trust[target] = max(0.0, st.session_state.trust[target] - drop_amount)
    # neighbor effect: reduce others slightly
    for n in st.session_state.nodes:
        if n != target:
            st.session_state.trust[n] = max(0.0, st.session_state.trust[n] - drop_amount*0.2)

def orchestrator_isolate(node):
    st.session_state.status[node] = "isolated"
    log(f"Orchestrator: Isolated {node} (firewall applied).")
    # start honeypot if not active
    st.session_state.honeypot_active = True
    log("Orchestrator: Deployed honeypot to capture attacker traffic.")

def orchestrator_reintegrate(node):
    st.session_state.status[node] = "trusted"
    st.session_state.trust[node] = min(0.95, st.session_state.trust[node] + 0.6)
    st.session_state.honeypot_active = False
    log(f"Orchestrator: Reintegrated {node} after remediation.")

# UI layout
col1, col2 = st.columns([1.4,1])

with col1:
    st.header("SCADA Mesh Visual")
    draw_mesh(st.session_state.trust)
    st.markdown("**Node status**")
    df = pd.DataFrame([
        {"node": n, "trust": round(st.session_state.trust[n],3), "status": st.session_state.status[n]}
        for n in st.session_state.nodes
    ])
    st.table(df.set_index("node"))

with col2:
    st.header("Controls & Attack")
    if not st.session_state.attack_active and st.button("▶ Start Attack"):
        st.session_state.attack_active = True
        log(f"Attacker: Started attack on {attack_node} (intensity={attack_strength}).")
    if st.session_state.attack_active and st.button("⏸ Stop Attack"):
        st.session_state.attack_active = False
        log("Attacker: Stopped attack.")
    if st.button("⚙️ Reset Demo"):
        # reset state
        st.session_state.trust = {n: 0.95 for n in st.session_state.nodes}
        st.session_state.status = {n: "trusted" for n in st.session_state.nodes}
        st.session_state.log = []
        st.session_state.attack_active = False
        st.session_state.honeypot_active = False
        st.experimental_rerun()

    st.markdown("---")
    st.subheader("Orchestrator")
    if st.button("Isolate selected node now"):
        orchestrator_isolate(attack_node)
    if st.button("Reintegrate selected node now"):
        orchestrator_reintegrate(attack_node)

    st.markdown("---")
    st.subheader("Honeypot")
    st.write("Honeypot active:" , "✅" if st.session_state.honeypot_active else "❌")
    if st.session_state.honeypot_active:
        st.info("Honeypot is capturing attacker requests. (Simulated)")

# background loop simulation zone (emulate ticks)
tick_col1, tick_col2 = st.columns([1,1])
with tick_col1:
    st.subheader("Live events / logs")
    if len(st.session_state.log) == 0:
        st.write("No events yet. Start attack or press Step.")
    else:
        # show last 8 logs
        for line in st.session_state.log[:12]:
            st.write(line)

with tick_col2:
    st.subheader("Attacker view (simulated)")
    if st.session_state.honeypot_active:
        st.write("Attacker is now interacting with a honeypot. Their commands are logged but real system is safe.")
    else:
        st.write("Attacker is targeting a real SCADA node.")

# auto-run mode or manual step
if auto_flow == "Auto run (recommended)":
    # run a compact auto loop of a few iterations (but Streamlit reruns, so we use timer + session flag)
    if "autorun_stage" not in st.session_state:
        st.session_state.autorun_stage = 0
        st.session_state.autorun_time = time.time()

    # run limited number of cycles only
    if st.session_state.autorun_stage < 12:
        # small delay so UI updates
        time.sleep(0.6)
        if st.session_state.attack_active == False:
            # start attack automatically at beginning
            st.session_state.attack_active = True
            log(f"[Auto] Attacker started on {attack_node}")
        # if attack active, compute anomaly
        if st.session_state.attack_active:
            score = anomaly_score(attack_node, attack_strength)
            st.session_state.attacker_history.append(score)
            log(f"Edge AI: Detected anomaly score {score:.2f} on {attack_node}")
            # if anomaly high and not isolated yet => orchestrate
            if score > 0.48 and st.session_state.status[attack_node] != "isolated":
                log(f"AI -> intent: quarantine {attack_node} (trust threshold breached).")
                # simulate GNN-propagation
                propagate_trust_drop(attack_node, drop_amount=min(0.7, 0.15 * attack_strength))
                orchestrator_isolate(attack_node)
            st.session_state.autorun_stage += 1
    else:
        # after cycles, auto reintegrate to show full cycle
        if st.session_state.status[attack_node] == "isolated":
            log("[Auto] Remediation simulated. Reintegrating node...")
            orchestrator_reintegrate(attack_node)
        # stop autorun after one full cycle
        st.session_state.autorun_stage = 999

else:
    # Manual stepping - user presses Step button
    if st.button("Step (manual tick)"):
        if st.session_state.attack_active:
            score = anomaly_score(attack_node, attack_strength)
            st.session_state.attacker_history.append(score)
            log(f"Edge AI: Detected anomaly score {score:.2f} on {attack_node}")
            if score > 0.48 and st.session_state.status[attack_node] != "isolated":
                log(f"AI -> intent: quarantine {attack_node} (trust threshold breached).")
                propagate_trust_drop(attack_node, drop_amount=min(0.7, 0.15 * attack_strength))
                orchestrator_isolate(attack_node)

# show trust score timeline chart (simple)
st.markdown("---")
st.subheader("Trust scores history (simulated)")
if len(st.session_state.attacker_history) > 0:
    hist_df = pd.DataFrame({
        "tick": list(range(1, len(st.session_state.attacker_history)+1)),
        "anomaly": st.session_state.attacker_history
    })
    st.line_chart(hist_df.set_index("tick"))
else:
    st.write("No anomaly data yet — start the attack.")

st.markdown("---")
st.caption("This interactive demo is a simulation: it visualizes the core Aegis-INTENT flow (edge AI detection, trust scoring, intent translation, orchestration of isolation/honeypot, and reintegration). The same logic applies to real SCADA/OT deployments with physical PLCs, RAN slices, and OSM orchestration.")
