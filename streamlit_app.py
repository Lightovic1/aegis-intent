# streamlit_app.py
# Aegis-INTENT Advanced Interactive Demo
# Designed for Streamlit Cloud (browser-only). No heavy ML libs required.
# Features:
# - multiple scenarios (real-case presets)
# - several attack types and configurable intensity
# - simulated edge AI anomaly detection, trust scoring, GNN propagation
# - policy editor (AIP simulation) + orchestrator (OIL simulation)
# - federated-learning progress visualization
# - logs, export, and guided presenter script to run a 10-15 min demo

import streamlit as st
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import time
from datetime import datetime
import json
import altair as alt
import io

st.set_page_config(layout="wide", page_title="Aegis-INTENT — Advanced Demo", page_icon="🛡️")

# -------------------------
# Helper / utility methods
# -------------------------
def nowstr():
    return datetime.utcnow().strftime("%H:%M:%S")

def init_state(num_nodes=5):
    nodes = [f"node{i}" for i in range(1, num_nodes+1)]
    st.session_state.nodes = nodes
    st.session_state.trust = {n: 0.95 for n in nodes}
    st.session_state.status = {n: "trusted" for n in nodes}
    st.session_state.log = []
    st.session_state.attack_active = False
    st.session_state.honeypot = {}
    st.session_state.orchestrator_actions = []
    st.session_state.attacker_history = []
    st.session_state.model_quality = 0.55
    st.session_state.scenario = "Custom"
    st.session_state.policy_text = "If trust < 0.4 then isolate node and deploy honeypot."
    st.session_state.step_index = 0

def log(msg):
    st.session_state.log.insert(0, f"[{nowstr()}] {msg}")

# --------------------------------
# Small deterministic "AI" models
# --------------------------------
def anomaly_score_attack(atype, intensity):
    # deterministic pseudo-random function for reproducibility
    base = {
        "noisy_toggle": 0.05 * intensity,
        "burst": 0.06 * intensity,
        "replay": 0.03 * intensity,
        "firmware_spoof": 0.08 * intensity,
        "stealth": 0.02 * intensity
    }.get(atype, 0.05 * intensity)
    # simulate effect of federated model quality reducing false positives
    quality = st.session_state.model_quality
    raw = np.tanh(base * 2.2)
    adjusted = raw * (1.0 - 0.15 * quality)
    return float(np.clip(adjusted, 0.0, 1.0))

def propagate_trust_gnn(target, strength):
    # simulate Graph Neural Network style propagation with adjacency ring + diagonals
    nodes = st.session_state.nodes
    N = len(nodes)
    # big drop at target, smaller for neighbors
    drop_target = min(0.9, 0.2 + 0.05 * strength)
    st.session_state.trust[target] = max(0.0, st.session_state.trust[target] - drop_target)
    for i, n in enumerate(nodes):
        if n == target: continue
        # distance effect: neighbors drop more
        idx_t = nodes.index(target)
        dist = min(abs(i - idx_t), N - abs(i - idx_t))
        attenuation = 0.7 ** dist
        st.session_state.trust[n] = max(0.0, st.session_state.trust[n] - drop_target * 0.3 * attenuation)

# -------------------------
# UI: left column - visual
# -------------------------
st.title("Aegis-INTENT — Advanced Interactive Demo 🛡️")
st.markdown("**Extended demo**: simulate attacks on a SCADA mesh and show how Aegis-INTENT components (Edge AI → DTEE → AIP/IPM → OIL/OSM → honeypot) act together.")

# top controls
with st.sidebar:
    st.header("Demo Controls")
    nodes_count = st.slider("Nodes in SCADA mesh", 3, 10, 5)
    if "nodes" not in st.session_state or len(st.session_state.nodes) != nodes_count:
        init_state(nodes_count)
    # Scenario presets
    st.subheader("Scenario Preset")
    scenario = st.selectbox("Choose preset", ["Custom (interactive)","Colonial Pipeline 2021","Oldsmar Water 2021","Ukraine Grid 2015","Triton 2017","Jaguar Land Rover 2025"], index=0)
    st.session_state.scenario = scenario
    st.subheader("Attack Configuration")
    attack_type = st.selectbox("Attack type", ["noisy_toggle","burst","replay","firmware_spoof","stealth"])
    attack_target = st.selectbox("Target node", st.session_state.nodes, index=0)
    attack_intensity = st.slider("Attack intensity", 1, 10, 6)
    attack_speed = st.slider("Attack cadence (ticks/sec)", 1, 4, 1)
    st.subheader("Policy / Intent")
    st.session_state.policy_text = st.text_area("Human intent (one line)", st.session_state.policy_text, height=80)
    st.button("Translate Intent (AIP)", on_click=lambda: log("[AIP] Translated intent -> " + translate_intent(st.session_state.policy_text)))
    st.markdown("---")
    st.subheader("Federated Learning (simulated)")
    st.session_state.model_quality = st.slider("Global model quality (0=poor,1=best)", 0.0, 0.95, st.session_state.model_quality)
    st.markdown("Higher model quality reduces false positives and improves detection.")
    st.markdown("---")
    st.subheader("Demo Flow")
    autorun = st.selectbox("Run mode", ["Manual step (use Step button)", "Auto (continuous)"], index=1)
    st.button("Reset demo", on_click=lambda: init_state(nodes_count))

# helper: translate intent (very small mock of AIP)
def translate_intent(text):
    text_low = text.lower()
    actions = []
    if "isolate" in text_low or "quarantine" in text_low:
        actions.append("isolate_node")
    if "honeypot" in text_low or "trap" in text_low:
        actions.append("deploy_honeypot")
    if "reroute" in text_low or "maintain" in text_low:
        actions.append("reroute_traffic")
    if "notify" in text_low or "alert" in text_low:
        actions.append("notify_ops")
    if len(actions) == 0:
        actions.append("log_only")
    return actions

# draw SCADA mesh graph
def draw_mesh(trust_map, highlight=None):
    G = nx.Graph()
    nodes = list(trust_map.keys())
    G.add_nodes_from(nodes)
    # ring and diagonals for a mesh-like look
    for i in range(len(nodes)):
        G.add_edge(nodes[i], nodes[(i+1) % len(nodes)])
    for i in range(len(nodes)//2):
        G.add_edge(nodes[i], nodes[(i+2) % len(nodes)])
    pos = nx.spring_layout(G, seed=42, k=0.7)
    fig, ax = plt.subplots(figsize=(7,4))
    colors = []
    sizes = []
    for n in nodes:
        t = trust_map[n]
        colors.append((1.0 - t, t*0.6, 0.1))
        sizes.append(300 * (0.5 + t))
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=sizes, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=9, ax=ax)
    nx.draw_networkx_edges(G, pos, alpha=0.4, ax=ax)
    if highlight:
        nx.draw_networkx_nodes(G, pos, nodelist=[highlight], node_color='cyan', node_size=500, node_shape='s', ax=ax)
    ax.set_title("SCADA Mesh (node color = trust)")
    ax.axis('off')
    st.pyplot(fig)
    plt.close(fig)

# layout: two main columns
col_left, col_mid, col_right = st.columns([1.1, 0.9, 0.9])

with col_left:
    st.header("Mesh Visualization")
    draw_mesh(st.session_state.trust, highlight=attack_target if st.session_state.attack_active else None)
    st.markdown("**Node table**")
    df_nodes = pd.DataFrame([{"node":n, "trust":round(st.session_state.trust[n],3), "status":st.session_state.status[n]} for n in st.session_state.nodes])
    st.table(df_nodes.set_index("node"))

with col_mid:
    st.header("Edge AI & Trust (DTEE)")
    st.write("Edge AI detects anomalies and DTEE combines signals into trust scores.")
    # trust heatmap
    trust_df = pd.DataFrame({"node":list(st.session_state.trust.keys()), "trust":[st.session_state.trust[n] for n in st.session_state.trust]})
    chart = alt.Chart(trust_df).mark_bar().encode(
        x=alt.X('node:N', sort=list(st.session_state.trust.keys())),
        y=alt.Y('trust:Q'),
        color=alt.Color('trust:Q', scale=alt.Scale(domain=[0,1], scheme='redyellowgreen'))
    ).properties(height=200)
    st.altair_chart(chart, use_container_width=True)
    st.markdown("**Anomaly timeline**")
    if len(st.session_state.attacker_history) > 0:
        histdf = pd.DataFrame({"tick": list(range(1,len(st.session_state.attacker_history)+1)), "anomaly": st.session_state.attacker_history})
        st.line_chart(histdf.set_index("tick"))
    else:
        st.write("No anomaly events yet. Run attack (manual step or Auto).")

    st.markdown("---")
    st.subheader("Policy (IPM / AIP)")
    st.write("Human intent (left) is translated by the AIP into actions that the orchestrator executes.")
    st.code(st.session_state.policy_text, language='text')
    st.write("Translated actions (AIP):", ", ".join(translate_intent(st.session_state.policy_text)))

    st.markdown("---")
    st.subheader("Orchestrator (OIL / OSM) actions")
    if len(st.session_state.orchestrator_actions) == 0:
        st.write("No orchestration actions yet.")
    else:
        for i, a in enumerate(st.session_state.orchestrator_actions[:8]):
            st.write(f"{i+1}. {a}")

with col_right:
    st.header("Attacker & Honeypot View")
    st.write("Preset scenario:", st.session_state.scenario)
    st.subheader("Attack controls")
    if not st.session_state.attack_active:
        if st.button("▶ Start Attack"):
            st.session_state.attack_active = True
            log(f"Attacker: start on {attack_target} (type={attack_type}, intensity={attack_intensity})")
    else:
        if st.button("⏸ Stop Attack"):
            st.session_state.attack_active = False
            log("Attacker: stopped by operator")

    if st.button("Step (manual tick)"):
        run_tick(attack_target, attack_type, attack_intensity)
    if autorun == "Auto (continuous)":
        # small auto loop trigger: run a few ticks then stop
        run_auto_ticks = 10
        for i in range(run_auto_ticks):
            run_tick(attack_target, attack_type, attack_intensity)
            time.sleep(0.5)

    st.markdown("---")
    st.subheader("Honeypot status")
    if st.session_state.honeypot.get(attack_target, False):
        st.success(f"Honeypot deployed to catch attacker on {attack_target}")
    else:
        st.write("No honeypot deployed yet.")

    st.markdown("---")
    st.subheader("Logs (latest first)")
    if len(st.session_state.log) == 0:
        st.write("No events yet.")
    else:
        for line in st.session_state.log[:20]:
            st.write(line)

# ---------------------
# Core tick & actions
# ---------------------
def run_tick(target, atype, intensity):
    # 1) Edge AI: compute anomaly
    sc = anomaly_score_attack(atype, intensity)
    st.session_state.attacker_history.append(sc)
    log(f"EdgeAI: anomaly score {sc:.2f} on {target}")
    # 2) DTEE: if score > threshold -> lower trust via GNN propagate
    threshold = 0.48 * (1.0 - st.session_state.model_quality * 0.2)  # better model = slightly lower false positives
    if sc > threshold:
        log(f"DTEE: trust threshold breached for {target} (score {sc:.2f} > {threshold:.2f})")
        propagate_trust_gnn(target, strength=intensity)
        # 3) IPM/AIP: translate policy or auto-intent
        actions = translate_intent(st.session_state.policy_text)
        # auto augment: if trust very low -> force isolation
        if st.session_state.trust[target] < 0.35 and "isolate_node" not in actions:
            actions.append("isolate_node")
        log(f"AIP: planned actions -> {actions}")
        # 4) Orchestrator executes actions
        perform_orchestration(target, actions)
    else:
        log(f"DTEE: anomaly below threshold ({sc:.2f} <= {threshold:.2f}); monitoring.")

def perform_orchestration(target, actions):
    for a in actions:
        if a == "isolate_node":
            st.session_state.status[target] = "isolated"
            st.session_state.orchestrator_actions.insert(0, f"Firewall applied to {target}")
            log(f"OIL: Firewall applied to {target}")
        if a == "deploy_honeypot":
            st.session_state.honeypot[target] = True
            st.session_state.orchestrator_actions.insert(0, f"Honeypot deployed beside {target}")
            log(f"OIL: Honeypot deployed to trap attacker around {target}")
        if a == "reroute_traffic":
            st.session_state.orchestrator_actions.insert(0, f"Traffic rerouted around {target}")
            log(f"OIL: Traffic rerouted to maintain service if {target} offline")
        if a == "notify_ops":
            st.session_state.orchestrator_actions.insert(0, f"Ops team notified about {target}")
            log(f"OIL: Ops notified about suspicious activity on {target}")
        if a == "log_only":
            st.session_state.orchestrator_actions.insert(0, f"Logged event for {target}")
            log(f"OIL: Event logged for {target}")

# ---------------------------
# Scenario presets (mapping)
# ---------------------------
def apply_preset(name):
    if name == "Colonial Pipeline 2021":
        st.session_state.policy_text = "If central IT unavailable, continue local safe pumping and isolate suspect nodes; deploy honeypot."
        st.session_state.model_quality = 0.4
        st.session_state.trust = {n:0.95 for n in st.session_state.nodes}
        log("Preset applied: Colonial Pipeline (IT->OT fallback)")
    elif name == "Oldsmar Water 2021":
        st.session_state.policy_text = "Block out-of-range actuator commands and require 2FA for chemical dosing overrides; deploy honeypot on remote sessions."
        st.session_state.model_quality = 0.55
        st.session_state.trust = {n:0.95 for n in st.session_state.nodes}
        log("Preset applied: Oldsmar Water (actuator protections)")
    elif name == "Ukraine Grid 2015":
        st.session_state.policy_text = "If abnormal breaker commands detected, isolate node and preserve N-1 redundancy; reroute flows."
        st.session_state.model_quality = 0.5
        st.session_state.trust = {n:0.95 for n in st.session_state.nodes}
        log("Preset applied: Ukraine Grid (N-1 redundancy)")
    elif name == "Triton 2017":
        st.session_state.policy_text = "Block unsigned firmware updates and isolate safety controllers; deploy deep forensic VNFs when SIS anomalies detected."
        st.session_state.model_quality = 0.45
        st.session_state.trust = {n:0.98 for n in st.session_state.nodes}
        log("Preset applied: Triton (SIS protection)")
    elif name == "Jaguar Land Rover 2025":
        st.session_state.policy_text = "If supply-chain IT is impacted, enable local production safe-mode and deploy local VNFs to decouple OT from central IT."
        st.session_state.model_quality = 0.6
        st.session_state.trust = {n:0.95 for n in st.session_state.nodes}
        log("Preset applied: JLR (IT->OT decoupling)")
    else:
        log("Custom scenario selected.")

# apply the preset if changed
if st.session_state.scenario != "Custom (interactive)":
    apply_preset(st.session_state.scenario)

# ---------------------
# Export logs / download
# ---------------------
def get_log_csv():
    df = pd.DataFrame({"time_msg": st.session_state.log})
    csv = df.to_csv(index=False)
    return csv

buf = io.StringIO(get_log_csv())
st.download_button("Download logs (CSV)", data=buf.getvalue(), file_name="aegis_intent_logs.csv")

# -----------------------
# Federated learning UI
# -----------------------
st.markdown("---")
st.header("Federated Learning (Simulated)")
st.write("This panel simulates how local edge agents improve a shared model quality over time. Slide the 'model quality' to show effects on detection sensitivity and false positive rate.")
colA, colB = st.columns(2)
with colA:
    st.metric("Global model quality", f"{st.session_state.model_quality:.2f}")
    st.progress(int(st.session_state.model_quality*100))
with colB:
    st.write("Effect: higher model quality reduces false positives and improves anomaly classification. Use as talking point to explain privacy-preserving FL in paper.")

# -----------------------
# Guided presenter script
# -----------------------
st.markdown("---")
st.header("Presenter Script — 12–15 minute demo flow (suggested)")
st.markdown("""
**Start (0:00–0:30)** — Quick intro slide (one sentence): "We simulate an AI-driven self-defending SCADA mesh (Aegis-INTENT) that detects, isolates, deceives, and heals."

**Phase 1 — Setup & Concepts (0:30–2:30)**  
- Show mesh visualization and node table.  
- Explain acronyms: Edge AI, DTEE (Dynamic Trust Enforcement Engine), AIP (AI-Enhanced Intent Processor), IPM (Intent Policy Manager), OIL/OSM (Orchestrator).  
- Map GUI panes to architecture.

**Phase 2 — Scenario & Attack (2:30–5:30)**  
- Choose a preset (e.g., Oldsmar). Explain the real-world problem.  
- Start attack (auto or manual). Walk audience through anomaly score lines appearing.

**Phase 3 — Contain & Orchestrate (5:30–9:00)**  
- When threshold breaches, show DTEE lowering trust and GNN propagation.  
- Highlight orchestrator actions (firewall/honeypot) and check logs.  
- Explain policy translation: show AIP translated actions.

**Phase 4 — Forensics & Recovery (9:00–11:00)**  
- Explain how honeypot traps attacker for forensics.  
- Simulate remediation (press Reintegrate manually or let reintegrate).  
- Show trust recovery and explain reintegrate safety checks.

**Phase 5 — Federated learning & wrap-up (11:00–12:30)**  
- Move the model quality slider to show how detection profile changes.  
- Summarize: Detect → Contain → Deceive → Heal. Explain where to implement in real infra and benefits.

**Optional Q&A demo continuation (12:30–15:00)**  
- Show another preset (Ukraine or Colonial) and run a short auto demo to show other use-cases.
""")

st.markdown("---")
st.caption("This demo is intentionally simulated to be fully browser-based and stable in live sessions. It maps to the design and modules in the Aegis-INTENT paper and is intended as a teaching & demonstration tool.")

# Ensure that main UI remains responsive after actions
