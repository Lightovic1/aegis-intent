# streamlit_app.py
# Aegis-INTENT — Advanced Conference Demo
# Browser-only simulation of Aegis-INTENT (SCADA Mesh + Edge AI + DTEE + AIP + OIL/OSM + NIDT)
# Supports Manual step-by-step mode (with animations/explanations) and Auto full-run.
# Paste this file into your GitHub repo and deploy to Streamlit Cloud.

import streamlit as st
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import time, json, io
from datetime import datetime
import altair as alt

st.set_page_config(layout="wide", page_title="Aegis-INTENT: Advanced Demo", page_icon="🛡️")

# ------------------------
# Utility helpers
# ------------------------
def now():
    return datetime.utcnow().strftime("%H:%M:%S")

def init_session(num_nodes=6):
    if "initialized" not in st.session_state:
        st.session_state.initialized = True
    st.session_state.nodes = [f"node{i}" for i in range(1, num_nodes+1)]
    st.session_state.trust = {n:0.96 for n in st.session_state.nodes}
    st.session_state.status = {n:"trusted" for n in st.session_state.nodes}
    st.session_state.honeypot = {n: False for n in st.session_state.nodes}
    st.session_state.orchestrator_log = []
    st.session_state.audit_log = []
    st.session_state.attack_active = False
    st.session_state.anomaly_history = []
    st.session_state.model_quality = 0.60
    st.session_state.current_scenario = "Custom"
    st.session_state.attack_info = {}
    st.session_state.manual_step = 0
    st.session_state.auto_running = False

def log_event(msg, level="INFO"):
    st.session_state.audit_log.insert(0, f"[{now()}] [{level}] {msg}")

def small_pause(sec=0.5):
    # makes the animation/view slow enough for demo; keep small.
    time.sleep(sec)

# ------------------------
# Scenarios & exact attack mapping (real incidents)
# ------------------------
SCENARIOS = {
    "Custom": {
        "title":"Custom scenario",
        "desc":"Interactive custom scenario. Choose attack type and parameters manually.",
        "attack_type":"noisy_toggle",
        "ioas": [],
        "iocs": []
    },
    "Colonial Pipeline 2021": {
        "title":"Colonial Pipeline (May 2021)",
        "desc":"Ransomware (DarkSide) disabled central IT; IT->OT coupling caused operational halt.",
        "attack_type":"ransomware_supplychain",  # simulated category
        "ioas":[ "Credential misuse", "Unauthorized VPN login", "Data encryption activity" ],
        "iocs":[ "Compromised VPN account", "Known DarkSide C2 indicators", "Unusual process creation" ]
    },
    "Oldsmar Water 2021": {
        "title":"Oldsmar Water Plant (Feb 2021)",
        "desc":"Attacker gained remote access and attempted to change chemical dosing.",
        "attack_type":"remote_access_actuator_override",
        "ioas":[ "Out-of-range actuator command", "Session takeover", "Mouse movement from remote IP" ],
        "iocs":[ "Remote control session from suspicious IP", "Attempted setpoint change 100x normal" ]
    },
    "Ukraine Grid 2015": {
        "title":"Ukraine Power Grid (Dec 2015)",
        "desc":"Coordinated ICS malware targeted breakers and operator systems causing large outages.",
        "attack_type":"coordinated_switching",
        "ioas":[ "Unauthorized breaker commands", "Lateral movement between substations" ],
        "iocs":[ "Phishing initial payload", "Malicious remote commands to breakers", "BlackEnergy artifacts" ]
    },
    "Triton 2017": {
        "title":"Triton/Trisis (2017)",
        "desc":"Malware targeting safety-instrumented systems (SIS) to disable safety controllers.",
        "attack_type":"targeted_sis_firmware_tamper",
        "ioas":[ "Unsigned firmware write", "Safety controller abnormal state", "Memory tampering signatures" ],
        "iocs":[ "Attempted Triconex access", "Malicious firmware write attempts" ]
    },
    "Jaguar Land Rover 2025": {
        "title":"Jaguar Land Rover (2025) — IT supply-chain disruption",
        "desc":"Large IT outage affected factory scheduling/supply chain; OT couldn't continue due to dependency.",
        "attack_type":"it_supplychain_cascade",
        "ioas":[ "ERP unavailability triggers production pause", "Inventory system failure", "Supplier API failures" ],
        "iocs":[ "Compromised ERP session tokens", "Massive API timeouts to suppliers" ]
    }
}

# ------------------------
# Attack behaviours (simulators)
# ------------------------
def compute_anomaly(attack_type, intensity, model_quality):
    # returns anomaly score 0..1 depending on attack and model accuracy
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
    # model_quality reduces false positives: higher quality reduces anomaly magnitude for benign noise
    # but for real attacks we keep anomaly high; simulate nuance:
    quality_factor = max(0.5, 1.0 - model_quality*0.4)  # better model reduces baseline noise
    raw = np.tanh(base * 2.2) * quality_factor
    # deterministic jitter for demo reproducibility
    return float(np.clip(raw + (0.02 * (np.sin(intensity + len(attack_type)) )), 0.0, 1.0))

def gnn_propagation(trust_map, target, severity):
    nodes = list(trust_map.keys())
    N = len(nodes)
    tgt_idx = nodes.index(target)
    drop_target = min(0.9, 0.25 + 0.03*severity)
    new_trust = trust_map.copy()
    new_trust[target] = max(0.0, trust_map[target] - drop_target)
    # neighbors drop in attenuated form
    for i, n in enumerate(nodes):
        if n == target: continue
        dist = min(abs(i - tgt_idx), N - abs(i - tgt_idx))
        attenuation = (0.6 ** dist)
        new_trust[n] = max(0.0, trust_map[n] - drop_target * 0.28 * attenuation)
    return new_trust

# ------------------------
# AIP: translate human intent to structured actions
# ------------------------
def aip_translate(intent_text):
    text = intent_text.lower()
    actions = []
    if any(k in text for k in ["isolate","quarantine","block"]):
        actions.append("isolate_node")
    if any(k in text for k in ["honeypot","trap","deceive"]):
        actions.append("deploy_honeypot")
    if any(k in text for k in ["reroute","route","maintain"]):
        actions.append("reroute_flows")
    if any(k in text for k in ["notify","alert"]):
        actions.append("notify_ops")
    if any(k in text for k in ["forensic","forensics","capture"]):
        actions.append("deploy_forensic_vnf")
    if len(actions)==0:
        actions.append("log_only")
    return actions

# ------------------------
# Orchestrator simulation (OIL)
# ------------------------
def orchestrator_execute(target, actions):
    for a in actions:
        if a == "isolate_node":
            st.session_state.status[target] = "isolated"
            st.session_state.orchestrator_log.insert(0, f"{now()} - Firewall applied to {target}")
            log_event(f"Orchestrator applied firewall to {target}")
        if a == "deploy_honeypot":
            st.session_state.honeypot[target] = True
            st.session_state.orchestrator_log.insert(0, f"{now()} - Honeypot deployed near {target}")
            log_event(f"Orchestrator deployed honeypot near {target}")
        if a == "reroute_flows":
            st.session_state.orchestrator_log.insert(0, f"{now()} - Traffic rerouted to preserve service around {target}")
            log_event(f"Orchestrator rerouted traffic to preserve service around {target}")
        if a == "notify_ops":
            st.session_state.orchestrator_log.insert(0, f"{now()} - Ops notified for {target}")
            log_event(f"Ops team notified for {target}")
        if a == "deploy_forensic_vnf":
            st.session_state.orchestrator_log.insert(0, f"{now()} - Forensic VNF deployed for deep capture at {target}")
            log_event(f"Forensic VNF deployed for {target}")
        if a == "log_only":
            st.session_state.orchestrator_log.insert(0, f"{now()} - Event logged for {target}")
            log_event(f"Event logged for {target}")

# ------------------------
# Forensics / IOC capture simulation
# ------------------------
def capture_forensics(target, attack_type, intensity):
    # produce some sample IOC/IOA strings depending on attack_type
    iocs = []
    ioas = []
    if attack_type == "remote_access_actuator_override":
        iocs = ["session_from_suspicious_ip:198.51.100.23", "setpoint_change:+10000ppm", "service:TeamViewer/RemoteDesktop"]
        ioas = ["out_of_range_actuator_command", "unexpected_remote_mouse_control", "single-session_command_burst"]
    elif attack_type == "coordinated_switching":
        iocs = ["malicious_command_seq:SWITCH_SEQ_1", "spearphish_email_payload_hash:abc123"]
        ioas = ["simultaneous_breaker_ops", "suspicious_lateral_movement"]
    elif attack_type == "targeted_sis_firmware_tamper":
        iocs = ["unsigned_firmware_blob:triconex_mod", "abnormal_memory_write"]
        ioas = ["safety_controller_memory_tamper", "unexpected_firmware_write"]
    elif attack_type == "ransomware_supplychain":
        iocs = ["suspicious_process:encryptor.exe", "c2_domain:darkside.example"]
        ioas = ["file_encryption_activity", "prior_credential_theft"]
    elif attack_type == "it_supplychain_cascade":
        iocs = ["erp_timeout:500ms", "supplier_api_500"]
        ioas = ["many_service_timeouts", "api_auth_failures"]
    else:
        iocs = ["unusual_modbus_write_sequence"]
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
    log_event(f"Forensics captured IOCs: {', '.join(iocs)} and IOAs: {', '.join(ioas)}", level="FORENSICS")
    return report

# ------------------------
# Reintegrate / remediation simulation
# ------------------------
def remediate_and_reintegrate(target):
    # simulate remediation steps: firmware attestation, rollback, restart
    log_event(f"Remediation: Running firmware attestation on {target}")
    small_pause(0.6)
    # assume remediation succeeds for demo
    st.session_state.trust[target] = min(0.95, st.session_state.trust[target] + 0.6)
    st.session_state.status[target] = "trusted"
    st.session_state.honeypot[target] = False
    orchestrator_execute(target, ["notify_ops"])
    log_event(f"{target} remediated and reintegrated. Trust restored to {st.session_state.trust[target]:.2f}")

# ------------------------
# Visual helpers
# ------------------------
def draw_mesh(trust_map, highlight=None):
    G = nx.Graph()
    nodes = list(trust_map.keys())
    G.add_nodes_from(nodes)
    for i in range(len(nodes)):
        G.add_edge(nodes[i], nodes[(i+1)%len(nodes)])
    pos = nx.spring_layout(G, seed=23)
    fig, ax = plt.subplots(figsize=(7,4))
    colors = []
    sizes = []
    for n in nodes:
        t = trust_map[n]
        colors.append((1.0 - t, t*0.6, 0.12))
        sizes.append(400*(0.6 + t))
    nx.draw_networkx_nodes(G,pos,node_size=sizes,node_color=colors,ax=ax)
    nx.draw_networkx_labels(G,pos,font_size=9,ax=ax)
    nx.draw_networkx_edges(G,pos,alpha=0.35,ax=ax)
    if highlight:
        nx.draw_networkx_nodes(G,pos,nodelist=[highlight],node_color='cyan',node_size=700,ax=ax)
    ax.set_title("SCADA Mesh (node color = trust)")
    ax.axis('off')
    st.pyplot(fig)
    plt.close(fig)

# ------------------------
# INITIALIZE
# ------------------------
if "initialized" not in st.session_state:
    init_session(6)

# ------------------------
# Sidebar controls
# ------------------------
with st.sidebar:
    st.title("Aegis-INTENT Controls")
    nodes_count = st.slider("Mesh nodes", 3, 12, len(st.session_state.nodes))
    if nodes_count != len(st.session_state.nodes):
        init_session(nodes_count)
    scenario = st.selectbox("Scenario preset", list(SCENARIOS.keys()), index=list(SCENARIOS.keys()).index(st.session_state.current_scenario) if st.session_state.current_scenario in SCENARIOS else 0)
    if scenario != st.session_state.current_scenario:
        st.session_state.current_scenario = scenario
        # apply preset settings
        preset = SCENARIOS[scenario]
        # set up model quality defaults and policy text
        if "model_quality" in preset:
            st.session_state.model_quality = preset["model_quality"]
        # set policy text examples for scenarios
        if scenario == "Oldsmar Water 2021":
            st.session_state.policy_text = "Block out-of-range actuator commands; require human 2FA for dosing overrides; deploy honeypot on remote sessions"
        elif scenario == "Colonial Pipeline 2021":
            st.session_state.policy_text = "If central IT unavailable, enable local safe pumping, isolate suspected nodes, deploy local service VNFs"
        elif scenario == "Ukraine Grid 2015":
            st.session_state.policy_text = "On abnormal breaker commands, isolate node and preserve N-1 redundancy; reroute flows"
        elif scenario == "Triton 2017":
            st.session_state.policy_text = "Block unsigned firmware writes to SIS; deploy deep forensic VNFs on SIS anomalies"
        elif scenario == "Jaguar Land Rover 2025":
            st.session_state.policy_text = "If supplier/ERP faults detected, enable local safe-mode, spin up local inventory cache VNFs and isolate compromised IT"
        else:
            st.session_state.policy_text = "If trust < 0.4 then isolate node and deploy honeypot."

    st.write("Scenario description:")
    st.info(SCENARIOS[st.session_state.current_scenario]["desc"])
    st.markdown("---")
    st.subheader("Attack configuration (mapped to scenario)")
    # map attack type automatically for scenario, but allow override
    default_atype = SCENARIOS[st.session_state.current_scenario].get("attack_type","noisy_toggle")
    attack_type = st.selectbox("Attack type (auto-chosen by scenario)", [ "noisy_toggle","burst","replay","firmware_spoof","stealth",
                                                                       "remote_access_actuator_override","coordinated_switching",
                                                                       "targeted_sis_firmware_tamper","ransomware_supplychain","it_supplychain_cascade"], index=0 if default_atype=="noisy_toggle" else ["noisy_toggle","burst","replay","firmware_spoof","stealth",
                                                                       "remote_access_actuator_override","coordinated_switching",
                                                                       "targeted_sis_firmware_tamper","ransomware_supplychain","it_supplychain_cascade"].index(default_atype))
    attack_target = st.selectbox("Target node", st.session_state.nodes, index=0)
    attack_intensity = st.slider("Attack intensity", 1, 10, 6)
    st.markdown("---")
    st.subheader("Model & Policy")
    st.session_state.model_quality = st.slider("Global model quality (simulated)", 0.0, 0.95, st.session_state.model_quality)
    st.text_area("Policy / Intent (edit & press Translate)", value=st.session_state.get("policy_text",""), key="policy_text", height=100)
    if st.button("Translate Intent (AIP)"):
        actions = aip_translate(st.session_state.policy_text)
        log_event(f"AIP translated intent to actions: {actions}")
        st.success(f"AIP -> actions: {actions}")
    st.markdown("---")
    st.subheader("Demo mode")
    demo_mode = st.radio("Mode", ["Manual (step-by-step)","Auto (single run)"], index=0)
    st.markdown("Manual mode exposes buttons for each internal step (detect, score, intent, orchestrate, forensic, remediate). Auto runs entire chain.")
    st.markdown("---")
    st.write("Presenter tips:")
    st.write("- Use Manual to explain each stage and click while narrating.")
    st.write("- Use Auto to run a full scenario (10–15 min narration).")

# ------------------------
# LAYOUT: main columns
# ------------------------
col1, col2, col3 = st.columns([1.1, 0.9, 0.9])

with col1:
    st.header("SCADA Mesh")
    draw_mesh(st.session_state.trust, highlight=attack_target if st.session_state.attack_active else None)
    st.caption("Node color = trust (green high, red low). Click Start/Stop attack to run simulation.")

with col2:
    st.header("DTEE / Edge AI")
    st.write("Model quality:", f"{st.session_state.model_quality:.2f}")
    st.write("Anomaly timeline (recent first):")
    if len(st.session_state.anomaly_history) == 0:
        st.write("No anomalies yet. Start an attack (Manual Step or Auto).")
    else:
        df_hist = pd.DataFrame({"tick": list(range(1,len(st.session_state.anomaly_history)+1)), "anomaly": st.session_state.anomaly_history})
        st.line_chart(df_hist.set_index("tick"))

    st.markdown("---")
    st.subheader("Policy / Intent Manager (IPM)")
    st.text_area("Policy (human intent)", value=st.session_state.policy_text, key="policy_text_area", height=120)
    if st.button("Translate now (AIP)"):
        actions = aip_translate(st.session_state.policy_text)
        st.success(f"AIP produced actions: {actions}")
        log_event(f"AIP produced actions {actions}")

    st.markdown("---")
    st.subheader("Orchestrator (OIL) actions")
    if len(st.session_state.orchestrator_log) == 0:
        st.write("No orchestration actions yet.")
    else:
        for i, entry in enumerate(st.session_state.orchestrator_log[:8]):
            st.write(entry)

with col3:
    st.header("Attacker / Honeypot / Logs")
    # attack control panel
    if not st.session_state.attack_active:
        if st.button("▶ Start Attack"):
            st.session_state.attack_active = True
            log_event(f"Attacker started on {attack_target} (type={attack_type}, intensity={attack_intensity})")
    else:
        if st.button("⏸ Stop Attack"):
            st.session_state.attack_active = False
            log_event("Attacker stopped by operator")

    st.markdown("**Manual step controls**")
    if demo_mode == "Manual (step-by-step)":
        # step buttons
        if st.button("Step 1 — Detect (Edge AI)"):
            # compute anomaly and show
            sc = compute_anomaly(attack_type, attack_intensity, st.session_state.model_quality)
            st.session_state.anomaly_history.insert(0, sc)
            log_event(f"Edge AI detected anomaly score {sc:.2f} at {attack_target}")
            st.success(f"Anomaly score {sc:.2f}")
            st.session_state.manual_step = 1
        if st.button("Step 2 — Score & DTEE"):
            if st.session_state.manual_step < 1:
                st.error("Run Detect first (Step 1).")
            else:
                last = st.session_state.anomaly_history[0] if len(st.session_state.anomaly_history)>0 else 0.0
                # threshold depends on model quality
                threshold = 0.48 * (1.0 - st.session_state.model_quality*0.2)
                log_event(f"DTEE evaluating score {last:.2f} vs threshold {threshold:.2f}")
                if last > threshold:
                    # propagate trust drop via GNN
                    st.session_state.trust = gnn_propagation(st.session_state.trust, attack_target, attack_intensity)
                    log_event(f"DTEE: trust for {attack_target} fell to {st.session_state.trust[attack_target]:.2f}")
                    st.success(f"Trust dropped for {attack_target} to {st.session_state.trust[attack_target]:.2f}")
                else:
                    st.info("No trust breach.")
                st.session_state.manual_step = 2
        if st.button("Step 3 — Intent (AIP/IPM)"):
            if st.session_state.manual_step < 2:
                st.error("Run DTEE (Step 2) first.")
            else:
                actions = aip_translate(st.session_state.policy_text)
                st.session_state.pending_actions = actions
                log_event(f"AIP created actions {actions}")
                st.success(f"AIP -> {actions}")
                st.session_state.manual_step = 3
        if st.button("Step 4 — Orchestrate (OIL)"):
            if st.session_state.manual_step < 3:
                st.error("Run Intent (Step 3) first.")
            else:
                orchestrator_execute(attack_target, st.session_state.pending_actions)
                st.success("Orchestrator executed actions. Check OIL log.")
                st.session_state.manual_step = 4
        if st.button("Step 5 — Forensics (capture IOCs/IOAs)"):
            if st.session_state.manual_step < 4:
                st.error("Run Orchestrate (Step 4) first.")
            else:
                rep = capture_forensics(attack_target, attack_type, attack_intensity)
                st.json(rep)
                st.session_state.manual_step = 5
        if st.button("Step 6 — Remediate & Reinstate"):
            if st.session_state.manual_step < 5:
                st.error("Run Forensics (Step 5) first.")
            else:
                remediate_and_reintegrate(attack_target)
                st.success(f"{attack_target} remediated & reintegrated.")
                st.session_state.manual_step = 6
    else:
        # Auto mode
        if st.button("▶ Run Auto Full Scenario"):
            st.session_state.auto_running = True
            # run whole chain with animation
            st.experimental_rerun()

    st.markdown("---")
    st.subheader("Detailed Logs / IOCs / IOAs")
    if len(st.session_state.audit_log)==0:
        st.write("No events yet.")
    else:
        for l in st.session_state.audit_log[:20]:
            st.write(l)

# ------------------------
# Handle auto run outside UI blocks (so functions are available)
# ------------------------
if demo_mode == "Auto (single run)" and st.session_state.auto_running:
    # run a full sequence with animations
    attack_target_local = attack_target
    attack_type_local = attack_type
    intensity_local = attack_intensity
    # start
    log_event("Auto-run started: full scenario")
    # Stage 1: Detect (3 ticks)
    for tick in range(3):
        sc = compute_anomaly(attack_type_local, intensity_local, st.session_state.model_quality)
        st.session_state.anomaly_history.insert(0, sc)
        log_event(f"[Auto] EdgeAI tick {tick+1}: anomaly {sc:.2f}")
        small_pause(0.5)
    # DTEE & GNN
    threshold = 0.48 * (1.0 - st.session_state.model_quality*0.2)
    last = st.session_state.anomaly_history[0]
    log_event(f"[Auto] DTEE evaluating last anomaly {last:.2f} vs threshold {threshold:.2f}")
    if last > threshold:
        st.session_state.trust = gnn_propagation(st.session_state.trust, attack_target_local, intensity_local)
        log_event(f"[Auto] Trust dropped for {attack_target_local} -> {st.session_state.trust[attack_target_local]:.2f}")
        small_pause(0.6)
        # Intent translation
        actions = aip_translate(st.session_state.policy_text)
        log_event(f"[Auto] AIP translated policy -> {actions}")
        small_pause(0.6)
        orchestrator_execute(attack_target_local, actions)
        small_pause(0.6)
        # Forensics
        rep = capture_forensics(attack_target_local, attack_type_local, intensity_local)
        small_pause(0.8)
        # Show a simulated forensic analysis animation (progress)
        with st.spinner("Running forensic analysis and sandboxing..."):
            small_pause(1.2)
        # Remediate & reintegrate
        remediate_and_reintegrate(attack_target_local)
        log_event("[Auto] Remediation completed and node reintegrated.")
    else:
        log_event("[Auto] No threshold breach detected; monitoring continues.")
    st.session_state.auto_running = False
    st.experimental_rerun()

# ------------------------
# Export incident report
# ------------------------
st.markdown("---")
st.header("Export / Download")
if st.button("Download incident report (JSON)"):
    out = {
        "scenario": st.session_state.current_scenario,
        "attack_info": st.session_state.attack_info,
        "audit_log": st.session_state.audit_log,
        "orchestrator_log": st.session_state.orchestrator_log,
        "trust_state": st.session_state.trust,
        "timestamp": now()
    }
    b = io.BytesIO(json.dumps(out, indent=2).encode("utf-8"))
    st.download_button(label="Click to download JSON", data=b, file_name="aegis_intent_incident.json", mime="application/json")
