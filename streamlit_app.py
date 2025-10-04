# streamlit_app.py
# Aegis-INTENT — Dark SOC Dashboard Demo
# Presented by Jaydeep Katariya and Mohini Sharma @ c0c0n 2025
# Browser-only Streamlit app — simulates full detect->isolate->forensic->heal cycle
# Purpose: Demonstration for conference (no real PLCs/networks used)

import streamlit as st
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import altair as alt
import time, json, io
from datetime import datetime

# ----------------------
# Page setup & styling
# ----------------------
st.set_page_config(page_title="Aegis-INTENT SOC Demo", layout="wide", page_icon="🛡️")
# Minimal dark styling via st.markdown
st.markdown(
    """
    <style>
    .stApp { background-color: #0b1220; color: #e6eef8; }
    .big-title {font-size:30px; font-weight:700;}
    .muted {color:#9fb0c9; font-size:14px;}
    .card {background:#071026; padding:10px; border-radius:8px; box-shadow: 0 2px 6px rgba(0,0,0,0.5);}
    .small {font-size:12px; color:#9fb0c9}
    .log {font-family:monospace; color:#bfe5ff; background:#02111a; padding:6px; border-radius:6px;}
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------------
# Utilities
# ----------------------
def now():
    return datetime.utcnow().strftime("%H:%M:%S")

def small_pause(sec=0.6):
    time.sleep(sec)

def log_event(msg, level="INFO"):
    entry = {"time": now(), "level": level, "msg": msg}
    st.session_state["audit_log"].insert(0, entry)

def init_state(num_nodes=6):
    """Initialize session state values (idempotent)."""
    if "initialized" not in st.session_state:
        st.session_state.initialized = True
    st.session_state.nodes = [f"node{i}" for i in range(1, num_nodes+1)]
    st.session_state.trust = {n: 0.96 for n in st.session_state.nodes}
    st.session_state.status = {n: "trusted" for n in st.session_state.nodes}
    st.session_state.honeypot = {n: False for n in st.session_state.nodes}
    st.session_state.audit_log = []
    st.session_state.orch_log = []
    st.session_state.anomaly_history = []
    st.session_state.model_quality = 0.60
    st.session_state.policy_text = "If trust < 0.4 then isolate node and deploy honeypot."
    st.session_state.current_scenario = "Custom"
    st.session_state.attack_active = False
    st.session_state.attack_type = "noisy_toggle"
    st.session_state.attack_target = st.session_state.nodes[0] if st.session_state.nodes else "node1"
    st.session_state.attack_intensity = 6
    st.session_state.auto_running = False
    st.session_state.step = 0
    st.session_state.attack_info = {}
    st.session_state.timeline_progress = 0

# Ensure initial values exist
if "initialized" not in st.session_state:
    init_state(6)

# ----------------------
# Scenario definitions (historical mapping)
# ----------------------
SCENARIOS = {
    "Custom": {
        "desc": "Interactive custom scenario. Choose attack type manually.",
        "attack": "noisy_toggle",
        "iocs": ["unusual_modbus_write_sequence"],
        "ioas": ["rapid_coil_toggle"]
    },
    "Colonial Pipeline 2021": {
        "desc": "Ransomware (DarkSide) affected IT; IT->OT coupling forced shutdown.",
        "attack": "ransomware_supplychain",
        "iocs": ["compromised_vpn_account", "darkside_c2_domains", "suspicious_process_encryptor"],
        "ioas": ["credential_misuse", "file_encryption_activity", "it_ot_dependency_failure"]
    },
    "Oldsmar Water 2021": {
        "desc": "Remote access was used to alter chemical dosing; operator intervention stopped disaster.",
        "attack": "remote_access_actuator_override",
        "iocs": ["remote_session_from_suspicious_ip", "out_of_range_setpoint_change"],
        "ioas": ["unauthorized_actuator_command", "session_takeover", "sudden_setpoint_spike"]
    },
    "Ukraine Grid 2015": {
        "desc": "Coordinated ICS attacks issued malicious breaker commands; caused outages.",
        "attack": "coordinated_switching",
        "iocs": ["malicious_breaker_commands", "phishing_initial_payload"],
        "ioas": ["simultaneous_breaker_ops", "lateral_movement"]
    },
    "Triton 2017": {
        "desc": "Targeted safety (SIS) controllers; attackers tried to disable safety systems.",
        "attack": "targeted_sis_firmware_tamper",
        "iocs": ["unsigned_firmware_blob", "abnormal_memory_write"],
        "ioas": ["safety_controller_memory_tamper", "unexpected_firmware_write"]
    },
    "Jaguar Land Rover 2025": {
        "desc": "IT supply-chain / ERP outage cascaded to OT operations and halted production.",
        "attack": "it_supplychain_cascade",
        "iocs": ["erp_session_compromise", "supplier_api_timeouts"],
        "ioas": ["erp_unavailability", "inventory_mismatch"]
    }
}

# ----------------------
# Core simulation functions
# ----------------------
def compute_anomaly(attack_type, intensity, model_quality):
    """Return a 0..1 anomaly score influenced by attack type and model quality."""
    base_map = {
        "noisy_toggle": 0.06,
        "burst": 0.07,
        "replay": 0.035,
        "firmware_spoof": 0.12,
        "stealth": 0.02,
        "ransomware_supplychain": 0.05,
        "remote_access_actuator_override": 0.13,
        "coordinated_switching": 0.10,
        "targeted_sis_firmware_tamper": 0.15,
        "it_supplychain_cascade": 0.04
    }
    base = base_map.get(attack_type, 0.05) * intensity
    quality_factor = max(0.55, 1.0 - model_quality * 0.45)  # better model reduces baseline noise
    raw = np.tanh(base * 2.0) * quality_factor
    jitter = 0.02 * (np.sin(intensity + len(attack_type)))
    return float(np.clip(raw + jitter, 0.0, 1.0))

def gnn_trust_update(trust_map, target, severity):
    """Simulate GNN propagation - return updated trust_map dict."""
    nodes = list(trust_map.keys())
    N = len(nodes)
    idx = nodes.index(target)
    drop = min(0.85, 0.22 + 0.035 * severity)
    new = trust_map.copy()
    new[target] = max(0.0, trust_map[target] - drop)
    for i, n in enumerate(nodes):
        if n == target: continue
        dist = min(abs(i - idx), N - abs(i - idx))
        attenuation = (0.6 ** dist)
        new[n] = max(0.0, trust_map[n] - drop * 0.28 * attenuation)
    return new

def aip_translate(intent_text):
    """Simple translator converting human intent into orchestrator actions."""
    text = intent_text.lower()
    actions = []
    if any(k in text for k in ["isolate","quarantine","block"]):
        actions.append("isolate_node")
    if any(k in text for k in ["honeypot","trap","deceive"]):
        actions.append("deploy_honeypot")
    if any(k in text for k in ["reroute","maintain"]):
        actions.append("reroute_flows")
    if any(k in text for k in ["notify","alert"]):
        actions.append("notify_ops")
    if any(k in text for k in ["forensic","capture"]):
        actions.append("deploy_forensic_vnf")
    if not actions:
        actions.append("log_only")
    return actions

def orchestrator_execute(target, actions):
    """Simulate orchestration: update status, logs and show actions executed."""
    for a in actions:
        if a == "isolate_node":
            st.session_state.status[target] = "isolated"
            st.session_state.orch_log.insert(0, f"{now()} - Firewall policy applied to {target}")
            log_event(f"Orchestrator: Firewall applied to {target}")
        elif a == "deploy_honeypot":
            st.session_state.honeypot[target] = True
            st.session_state.orch_log.insert(0, f"{now()} - Honeypot deployed close to {target}")
            log_event(f"Orchestrator: Honeypot deployed near {target}")
        elif a == "reroute_flows":
            st.session_state.orch_log.insert(0, f"{now()} - Traffic rerouted to preserve service around {target}")
            log_event(f"Orchestrator: Rerouted traffic around {target}")
        elif a == "notify_ops":
            st.session_state.orch_log.insert(0, f"{now()} - Ops team notified about {target}")
            log_event(f"Orchestrator: Ops notified about {target}")
        elif a == "deploy_forensic_vnf":
            st.session_state.orch_log.insert(0, f"{now()} - Forensic VNF deployed for deep capture at {target}")
            log_event(f"Orchestrator: Forensic VNF deployed for {target}")
        else:
            st.session_state.orch_log.insert(0, f"{now()} - Logged event for {target}")
            log_event(f"Orchestrator: Event logged for {target}")

def capture_forensics(target, attack_type, intensity):
    """Simulate collection of IOCs and IOAs."""
    if attack_type == "remote_access_actuator_override":
        iocs = ["remote_session:198.51.100.23", "setpoint_change:+10000ppm", "service:remote_desktop"]
        ioas = ["out_of_range_actuator_command", "session_takeover", "sudden_setpoint_spike"]
    elif attack_type == "coordinated_switching":
        iocs = ["breaker_cmd_seq:SW-SEQ-1", "malicious_payload_hash:abc123"]
        ioas = ["simultaneous_breaker_ops", "lateral_movement"]
    elif attack_type == "targeted_sis_firmware_tamper":
        iocs = ["unsigned_fw_blob:triconex_mod", "memory_write_anomaly"]
        ioas = ["safety_controller_tamper", "unexpected_firmware_write"]
    elif attack_type == "ransomware_supplychain":
        iocs = ["suspicious_process:encryptor.exe", "c2_domain:darkside.example"]
        ioas = ["file_encryption_activity", "credential_theft"]
    elif attack_type == "it_supplychain_cascade":
        iocs = ["erp_token_compromise", "supplier_api_500s"]
        ioas = ["erp_unavailable", "supplychain_api_failures"]
    else:
        iocs = ["rapid_modbus_writes"]
        ioas = ["rapid_coil_toggle"]
    report = {
        "target": target,
        "attack_type": attack_type,
        "intensity": intensity,
        "iocs": iocs,
        "ioas": ioas,
        "timestamp": now()
    }
    st.session_state.attack_info = report
    log_event(f"Forensics: captured IOCs [{', '.join(iocs)}]", level="FORENSICS")
    return report

def remediate(target):
    """Simulate remediation & reintegration flow."""
    log_event(f"Remediation started for {target}")
    small_pause(0.8)
    # sample remediation: firmware attestation, config rollback
    st.session_state.trust[target] = min(0.95, st.session_state.trust[target] + 0.6)
    st.session_state.status[target] = "trusted"
    st.session_state.honeypot[target] = False
    orchestrator_execute(target, ["notify_ops"])
    log_event(f"Remediation completed for {target}; trust restored {st.session_state.trust[target]:.2f}")

# ----------------------
# Visualization helpers
# ----------------------
def draw_mesh(trust_map, highlight=None):
    G = nx.Graph()
    nodes = list(trust_map.keys())
    G.add_nodes_from(nodes)
    for i in range(len(nodes)):
        G.add_edge(nodes[i], nodes[(i+1) % len(nodes)])
    pos = nx.spring_layout(G, seed=42)
    fig, ax = plt.subplots(figsize=(6.8,4))
    colors = []
    sizes = []
    for n in nodes:
        t = trust_map[n]
        colors.append((1.0 - t, t*0.65, 0.08))
        sizes.append(300*(0.6 + t))
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=sizes, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=9, font_color='white', ax=ax)
    nx.draw_networkx_edges(G, pos, alpha=0.35, ax=ax)
    if highlight:
        nx.draw_networkx_nodes(G, pos, nodelist=[highlight], node_color='cyan', node_size=700, ax=ax)
    ax.set_facecolor("#071631")
    fig.patch.set_facecolor("#071631")
    ax.axis('off')
    st.pyplot(fig)
    plt.close(fig)

def trust_bar_chart(trust_map):
    df = pd.DataFrame({"node": list(trust_map.keys()), "trust": [trust_map[n] for n in trust_map]})
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X('node:N', sort=list(trust_map.keys())),
        y=alt.Y('trust:Q', scale=alt.Scale(domain=[0,1])),
        color=alt.Color('trust:Q', scale=alt.Scale(domain=[0,1], scheme='redyellowgreen'))
    ).properties(height=160)
    st.altair_chart(chart, use_container_width=True)

# ----------------------
# Top Header & Demo Script (always visible)
# ----------------------
st.markdown(f"<div class='big-title'>Presented by Jaydeep Katariya and Mohini Sharma @ c0c0n 2025</div>", unsafe_allow_html=True)
st.markdown("<div class='muted'>Aegis-INTENT: AI-Driven Cyber-Resilient SCADA Mesh in Open RAN-Enabled Smart Infrastructure — demo simulates detection, trust, intent, orchestration, and self-healing.</div>", unsafe_allow_html=True)

with st.container():
    st.markdown("### Demonstration Script (paper-aligned) — what you'll see in this demo")
    st.markdown("""
    1. **Detect** — Edge AI agents monitor PLC/RTU/SCADA telemetry and compute anomaly scores.  
    2. **Trust (DTEE)** — The Dynamic Trust Enforcement Engine fuses signals (behavior, firmware attestation, intel) and computes a node trust score.  
    3. **Intent (AIP/IPM)** — Human intent or AI-generated intent is translated to machine actions (isolate, honeypot, reroute).  
    4. **Orchestrate (OIL/OSM)** — Orchestrator deploys VNFs/CNFs (firewall, honeypot, forensic VNFs) to contain the threat.  
    5. **Forensics** — Honeypot & forensic VNFs capture IOCs/IOAs for investigation.  
    6. **Remediate & Reinstate** — System applies fixes (attestation, rollback) and safely reintegrates the node.  
    """)

# Attack timeline progress bar area
timeline_placeholder = st.empty()
timeline_col1, timeline_col2 = st.columns([2,1])
with timeline_col1:
    st.markdown("**Attack timeline**")
    progress = st.progress(0)
with timeline_col2:
    st.markdown("**Mode**")
    mode = st.radio("", ["Manual (step-by-step)","Auto (single-run)"], index=0, horizontal=True)

# ----------------------
# Sidebar controls (scenario, attack params, policy)
# ----------------------
with st.sidebar:
    st.header("Controls")
    nodes_count = st.slider("Mesh nodes", 3, 10, len(st.session_state.nodes))
    if nodes_count != len(st.session_state.nodes):
        init_state(nodes_count)
    scenario = st.selectbox("Scenario preset", list(SCENARIOS.keys()), index=list(SCENARIOS.keys()).index(st.session_state.current_scenario) if st.session_state.current_scenario in SCENARIOS else 0)
    if scenario != st.session_state.current_scenario:
        st.session_state.current_scenario = scenario
        preset = SCENARIOS[scenario]
        # apply preset attack mapping & policy text
        st.session_state.attack_type = preset.get("attack", "noisy_toggle")
        st.session_state.policy_text = "If trust < 0.4 then isolate node and deploy honeypot." if scenario=="Custom" else st.session_state.policy_text
        # inject a few lines to describe the scenario in logs
        log_event(f"Preset loaded: {scenario} - {preset['desc']}")
    st.markdown("**Scenario details**")
    st.info(SCENARIOS[st.session_state.current_scenario]["desc"])
    st.markdown("---")
    st.subheader("Attack configuration")
    # if the preset uses a mapped attack, default it but allow override
    attack_choice = st.selectbox("Attack type", ["noisy_toggle","burst","replay","firmware_spoof","stealth",
                                                "remote_access_actuator_override","coordinated_switching",
                                                "targeted_sis_firmware_tamper","ransomware_supplychain","it_supplychain_cascade"], index=0)
    st.session_state.attack_type = attack_choice
    target = st.selectbox("Target node", st.session_state.nodes, index=0)
    st.session_state.attack_target = target
    intensity = st.slider("Attack intensity", 1, 10, int(st.session_state.attack_intensity))
    st.session_state.attack_intensity = intensity
    st.markdown("---")
    st.subheader("Model & Intent")
    st.session_state.model_quality = st.slider("Global model quality (simulated)", 0.0, 0.95, float(st.session_state.model_quality))
    st.text_area("Policy / Intent", value=st.session_state.policy_text, key="policy_area", height=120)
    if st.button("Translate Intent (AIP)"):
        actions = aip_translate(st.session_state.policy_text)
        st.success(f"AIP -> actions: {actions}")
        log_event(f"AIP translated current policy into actions: {actions}")
    st.markdown("---")
    st.write("Presenter tip: Use Manual mode to explain each step. Use Auto mode for a single-run full scenario.")

# ----------------------
# Main 3-panel layout (SOC)
# ----------------------
left_col, mid_col, right_col = st.columns([1.1, 0.9, 0.95])

with left_col:
    st.subheader("Threat Map (SCADA Mesh)")
    draw_mesh(st.session_state.trust, highlight=st.session_state.attack_target if st.session_state.attack_active else None)
    st.markdown("<div class='small'>Node color = trust (green high → red low). Highlight = current attack target.</div>", unsafe_allow_html=True)
    trust_bar_chart(st.session_state.trust)

with mid_col:
    st.subheader("Edge AI & DTEE")
    st.markdown("<div class='small'>Shows anomaly timeline and model quality impact (simulated).</div>", unsafe_allow_html=True)
    st.metric("Global model quality", f"{st.session_state.model_quality:.2f}")
    # anomaly timeline chart
    if st.session_state.anomaly_history:
        hist_df = pd.DataFrame({"tick": list(range(1,len(st.session_state.anomaly_history)+1)), "anomaly": list(reversed(st.session_state.anomaly_history))})
        st.line_chart(hist_df.set_index("tick"))
    else:
        st.write("No anomalies yet. Start an attack to populate timeline.")
    st.markdown("---")
    st.subheader("DTEE — Trust Scores")
    st.write("Current trust scores (table):")
    df = pd.DataFrame([{"node": n, "trust": round(st.session_state.trust[n],3), "status": st.session_state.status[n]} for n in st.session_state.nodes])
    st.table(df.set_index("node"))

with right_col:
    st.subheader("Incident Console")
    st.markdown("<div class='small'>Logs, IOCs/IOAs and forensic reports appear here.</div>", unsafe_allow_html=True)
    # control buttons in the console
    if not st.session_state.attack_active:
        if st.button("▶ Start Attack (simulate)"):
            st.session_state.attack_active = True
            log_event(f"Attack started on {st.session_state.attack_target} type={st.session_state.attack_type} intensity={st.session_state.attack_intensity}")
    else:
        if st.button("⏸ Stop Attack"):
            st.session_state.attack_active = False
            log_event("Attack stopped by operator")

       # --------------------------
    # Manual mode controls (full 6-step flow)
    # --------------------------
    if mode == "Manual (step-by-step)":
        colA, colB = st.columns([1,1])
        with colA:
            # Step 1: Detect
            if st.button("Step 1 — Detect (Edge AI)"):
                sc = compute_anomaly(st.session_state.attack_type, st.session_state.attack_intensity, st.session_state.model_quality)
                st.session_state.anomaly_history.insert(0, sc)
                log_event(f"Edge AI: anomaly score {sc:.2f} on {st.session_state.attack_target}")
                st.session_state.step = 1
                # update timeline bar
                st.session_state.timeline_progress = 10
                progress.progress(st.session_state.timeline_progress)
                st.success(f"Detect complete: anomaly score {sc:.2f}")

            # Step 3: Intent (AIP) - placed in left column for spacing
            if st.button("Step 3 — Intent (AIP / IPM)"):
                if st.session_state.step < 2:
                    st.error("Run Step 2 — Score (DTEE) before step 3.")
                else:
                    actions = aip_translate(st.session_state.policy_text)
                    st.session_state.pending_actions = actions
                    log_event(f"AIP: translated intent -> {actions}")
                    st.session_state.step = 3
                    st.session_state.timeline_progress = 45
                    progress.progress(st.session_state.timeline_progress)
                    st.success(f"AIP created actions: {actions}")

        with colB:
            # Step 2: Score / DTEE
            if st.button("Step 2 — Score (DTEE)"):
                if st.session_state.step < 1:
                    st.error("Run Step 1 — Detect first.")
                else:
                    last = st.session_state.anomaly_history[0] if st.session_state.anomaly_history else 0.0
                    threshold = 0.48 * (1.0 - st.session_state.model_quality * 0.2)
                    log_event(f"DTEE: Evaluating last score {last:.2f} vs threshold {threshold:.2f}")
                    if last > threshold:
                        st.session_state.trust = gnn_trust_update(st.session_state.trust, st.session_state.attack_target, st.session_state.attack_intensity)
                        log_event(f"DTEE: trust for {st.session_state.attack_target} -> {st.session_state.trust[st.session_state.attack_target]:.2f}")
                        st.success(f"Trust drop: {st.session_state.attack_target} -> {st.session_state.trust[st.session_state.attack_target]:.2f}")
                    else:
                        st.info("No trust breach; continuing monitoring.")
                    st.session_state.step = 2
                    st.session_state.timeline_progress = 30
                    progress.progress(st.session_state.timeline_progress)

            # Step 4: Orchestrate (OIL)
            if st.button("Step 4 — Orchestrate (OIL)"):
                if st.session_state.step < 3:
                    st.error("Run Intent (Step 3) first.")
                else:
                    actions = st.session_state.get("pending_actions", aip_translate(st.session_state.policy_text))
                    orchestrator_execute(st.session_state.attack_target, actions)
                    log_event(f"OIL: executed actions {actions} on {st.session_state.attack_target}")
                    st.session_state.step = 4
                    st.session_state.timeline_progress = 60
                    progress.progress(st.session_state.timeline_progress)
                    st.success("Orchestrator executed actions. Check Orchestration log.")

        # Additional steps span full width (below the two columns)
        down_col1, down_col2 = st.columns([1,1])
        with down_col1:
            # Step 5: Forensics
            if st.button("Step 5 — Forensics (Capture IOCs/IOAs)"):
                if st.session_state.step < 4:
                    st.error("Run Orchestrate (Step 4) first.")
                else:
                    report = capture_forensics(st.session_state.attack_target, st.session_state.attack_type, st.session_state.attack_intensity)
                    st.session_state.attack_info = report
                    log_event(f"Forensics captured for {st.session_state.attack_target}: IOCs {report['iocs']}")
                    st.session_state.step = 5
                    st.session_state.timeline_progress = 80
                    progress.progress(st.session_state.timeline_progress)
                    st.success("Forensics capture complete. Review forensic report below.")
                    st.json(report)

        with down_col2:
            # Step 6: Remediate & Reinstate
            if st.button("Step 6 — Remediate & Reinstate"):
                if st.session_state.step < 5:
                    st.error("Run Forensics (Step 5) first.")
                else:
                    remediate(st.session_state.attack_target)
                    st.session_state.step = 6
                    st.session_state.timeline_progress = 100
                    progress.progress(st.session_state.timeline_progress)
                    st.success(f"{st.session_state.attack_target} remediated and reintegrated. Trust restored: {st.session_state.trust[st.session_state.attack_target]:.2f}")

    # End of Manual mode controls

    else:
        # Auto single-run execution (blocking but short)
        if st.button("▶ Run Full Scenario (Auto)"):
            st.session_state.auto_running = True
            log_event("Auto-run begun: executing full scenario")
            # tiny sequence of stages with pauses (for presentation)
            # Stage 1: Detect ticks
            for tick in range(3):
                sc = compute_anomaly(st.session_state.attack_type, st.session_state.attack_intensity, st.session_state.model_quality)
                st.session_state.anomaly_history.insert(0, sc)
                log_event(f"Edge AI [auto] tick {tick+1}: anomaly {sc:.2f}")
                st.session_state.timeline_progress = int(5 + tick*5)
                progress.progress(st.session_state.timeline_progress)
                small_pause(0.6)
            # Stage 2: DTEE evaluation
            last = st.session_state.anomaly_history[0]
            threshold = 0.48 * (1.0 - st.session_state.model_quality * 0.2)
            log_event(f"DTEE [auto] evaluating last {last:.2f} vs threshold {threshold:.2f}")
            small_pause(0.5)
            if last > threshold:
                # apply gnn propagation
                st.session_state.trust = gnn_trust_update(st.session_state.trust, st.session_state.attack_target, st.session_state.attack_intensity)
                log_event(f"DTEE [auto]: trust dropped for {st.session_state.attack_target} -> {st.session_state.trust[st.session_state.attack_target]:.2f}")
                st.session_state.timeline_progress = 45
                progress.progress(st.session_state.timeline_progress)
                small_pause(0.7)
                # Intent/AIP translation
                actions = aip_translate(st.session_state.policy_text)
                log_event(f"AIP [auto] translated policy to actions: {actions}")
                small_pause(0.5)
                st.session_state.timeline_progress = 55
                progress.progress(st.session_state.timeline_progress)
                # Orchestrator
                orchestrator_execute(st.session_state.attack_target, actions)
                st.session_state.timeline_progress = 70
                progress.progress(st.session_state.timeline_progress)
                small_pause(0.6)
                # Forensics
                report = capture_forensics(st.session_state.attack_target, st.session_state.attack_type, st.session_state.attack_intensity)
                st.session_state.timeline_progress = 82
                progress.progress(st.session_state.timeline_progress)
                small_pause(0.8)
                # Remediate & reintegrate
                remediate(st.session_state.attack_target)
                st.session_state.timeline_progress = 100
                progress.progress(st.session_state.timeline_progress)
                small_pause(0.6)
                log_event("Auto-run completed: node remediated and reintegrated")
            else:
                log_event("Auto-run: no threshold breach detected; monitoring")
            st.session_state.auto_running = False

    # Show logs & forensic details
    st.markdown("**Audit & Forensic Logs (latest first)**")
    if st.session_state.audit_log:
        for entry in st.session_state.audit_log[:18]:
            ts = entry["time"]
            lvl = entry["level"]
            msg = entry["msg"]
            color = "#9fb0c9" if lvl=="INFO" else "#ffd27f" if lvl=="FORENSICS" else "#ff9f9f"
            st.markdown(f"<div class='log'>[{ts}] <b>{lvl}</b> — {msg}</div>", unsafe_allow_html=True)
    else:
        st.write("No events yet. Start an attack to produce logs.")

# ----------------------
# Bottom: detailed forensic report & download
# ----------------------
st.markdown("---")
st.subheader("Forensic Report & Download")
if st.session_state.attack_info:
    st.json(st.session_state.attack_info)
else:
    st.info("Run forensics via Auto-run or manual Step 5 to populate forensic report.")

if st.button("Download Incident Report (JSON)"):
    out = {
        "presenters": "Jaydeep Katariya and Mohini Sharma",
        "conference": "c0c0n 2025",
        "timestamp": now(),
        "scenario": st.session_state.current_scenario,
        "attack": {
            "type": st.session_state.attack_type,
            "target": st.session_state.attack_target,
            "intensity": st.session_state.attack_intensity
        },
        "trust": st.session_state.trust,
        "status": st.session_state.status,
        "audit_log": st.session_state.audit_log,
        "orchestrator_log": st.session_state.orch_log,
        "forensics": st.session_state.attack_info
    }
    b = io.BytesIO(json.dumps(out, indent=2).encode("utf-8"))
    st.download_button("Download JSON", data=b, file_name="aegis_intent_incident.json", mime="application/json")

st.markdown("<div class='muted'>Tip: Use Manual mode to explain internals slowly. Use Auto mode to show the entire flow end-to-end in ~8-12 minutes, pausing between stages to narrate.</div>", unsafe_allow_html=True)
