# frontend/app.py

import streamlit as st
import requests
import json
import pandas as pd
import networkx as nx
import plotly.graph_objects as go

# --- Page Configuration ---
st.set_page_config(
    page_title="AML Case Management",
    page_icon="⚖️",
    layout="wide"
)

# --- API URLs ---
PREDICT_API_URL = "http://127.0.0.1:8000/predict"
GRAPH_API_URL = "http://127.0.0.1:8000/graph_analysis/"
CASES_API_URL = "http://127.0.0.1:8000/cases"

# --- Helper Functions ---
def display_shap_chart(explanation):
    """Helper function to display the SHAP bar chart."""
    df_exp = pd.DataFrame(explanation.items(), columns=['Feature', 'Contribution'])
    df_exp_pos = df_exp[df_exp['Contribution'] > 0].set_index('Feature')
    if not df_exp_pos.empty:
        st.bar_chart(df_exp_pos)

# --- Main App Logic ---
st.sidebar.title("AML Dashboard Navigation")
page = st.sidebar.radio("Go to", ["Live Transaction Analysis", "Case Management"])

if "last_analysis" not in st.session_state:
    st.session_state.last_analysis = None

# =====================================================================================
# --- PAGE 1: LIVE TRANSACTION ANALYSIS ---
# =====================================================================================
if page == "Live Transaction Analysis":
    st.title("Live Transaction Analysis 🕵️")
    col1, col2 = st.columns([1, 1.5])

    # --- Column 1: Single Transaction Analysis ---
    with col1:
        st.header("Check a Transaction")
        with st.form("transaction_form"):
            amount = st.number_input("Transaction Amount", min_value=0.0, format="%.2f")
            sender_age = st.number_input("Sender's Account Age (days)", min_value=1, step=1)
            receiver_age = st.number_input("Receiver's Account Age (days)", min_value=1, step=1)
            submitted = st.form_submit_button("Analyze Transaction")

        if submitted and amount > 0:
            payload = {"amount": amount, "sender_account_age": sender_age, "receiver_account_age": receiver_age}
            try:
                response = requests.post(PREDICT_API_URL, json=payload)
                response.raise_for_status()
                st.session_state.last_analysis = response.json()
            except requests.exceptions.RequestException as e:
                st.error(f"API Connection Error: {e}")
        
        if st.session_state.last_analysis:
            result = st.session_state.last_analysis
            st.subheader("Analysis Result")
            if result.get("is_fraud"):
                st.error("🚩 High Risk: Flagged as potentially fraudulent.")
                if st.button("Create Case from this Alert"):
                    case_payload = {
                        "transaction_details": {"amount": amount, "sender_account_age": sender_age, "receiver_account_age": receiver_age},
                        "risk_score": result.get('fraud_probability', 0),
                        "explanation": result.get('explanation', {})
                    }
                    try:
                        case_response = requests.post(CASES_API_URL, json=case_payload)
                        case_response.raise_for_status()
                        st.success(f"Case created successfully! Case ID: {case_response.json()['_id']}")
                    except requests.exceptions.RequestException as e:
                        st.error(f"Failed to create case: {e}")
            else:
                st.success("✅ Low Risk: Appears normal.")
            
            st.metric(label="Fraud Probability Score", value=f"{result.get('fraud_probability', 0):.2%}")
            if 'explanation' in result:
                st.write("**Risk Factors:**")
                display_shap_chart(result['explanation'])

    # --- Column 2: Graph Investigation ---
    with col2:
        st.header("Graph Investigation 🌐")
        with st.form("graph_form"):
            account_id_to_investigate = st.text_input("Account ID to Investigate", placeholder="e.g., 8799")
            graph_submitted = st.form_submit_button("Investigate Account")

        if graph_submitted and account_id_to_investigate:
            try:
                graph_response = requests.get(f"{GRAPH_API_URL}{account_id_to_investigate}")
                graph_response.raise_for_status()
                graph_data = graph_response.json()
                nodes = graph_data.get("nodes", [])
                edges = graph_data.get("edges", [])
                if not nodes:
                    st.warning("No transaction data found for this account.")
                else:
                    # Create a network graph visualization
                    G = nx.DiGraph()
                    
                    # Add nodes and edges
                    for node in nodes:
                        G.add_node(node["id"])
                    
                    for edge in edges:
                        G.add_edge(edge["source"], edge["target"], label=edge["label"])
                    
                    # Create positions for nodes using spring layout
                    pos = nx.spring_layout(G)
                    
                    # Create plotly figure
                    fig = go.Figure()
                    
                    # Add edges as annotations
                    for edge in G.edges(data=True):
                        source, target, data = edge
                        x0, y0 = pos[source]
                        x1, y1 = pos[target]
                        
                        fig.add_trace(
                            go.Scatter(
                                x=[x0, x1, None],
                                y=[y0, y1, None],
                                mode="lines+text",
                                line=dict(width=1, color="blue"),
                                text=["", data.get("label", "")],
                                textposition="middle center",
                                hoverinfo="none"
                            )
                        )
                    
                    # Add nodes
                    node_x = []
                    node_y = []
                    node_text = []
                    for node in G.nodes():
                        x, y = pos[node]
                        node_x.append(x)
                        node_y.append(y)
                        node_text.append(str(node))
                    
                    fig.add_trace(
                        go.Scatter(
                            x=node_x,
                            y=node_y,
                            mode="markers+text",
                            marker=dict(
                                size=25, 
                                color=["red" if node == account_id_to_investigate else "skyblue" for node in G.nodes()]
                            ),
                            text=node_text,
                            textposition="middle center",
                            hoverinfo="text"
                        )
                    )
                    
                    fig.update_layout(
                        title=f"Transaction Network for Account {account_id_to_investigate}",
                        showlegend=False,
                        margin=dict(l=0, r=0, b=0, t=40),
                        height=500
                    )
                    
                    # Display the graph
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display any detected circular patterns
                    circular_patterns = graph_data.get("patterns", {}).get("circular_transfers", [])
                    if circular_patterns:
                        st.subheader("⚠️ Suspicious Circular Transfers Detected")
                        for i, pattern in enumerate(circular_patterns):
                            st.write(f"**Circular Pattern {i+1}**: {' → '.join(str(p) for p in pattern)} → {pattern[0]}")

            except requests.exceptions.RequestException as e:
                st.error(f"API Connection Error: {e}")


# =====================================================================================
# --- PAGE 2: CASE MANAGEMENT ---
# =====================================================================================
elif page == "Case Management":
    st.title("Case Management Queue ⚖️")
    
    try:
        response = requests.get(CASES_API_URL)
        response.raise_for_status()
        cases = response.json()
        
        if not cases:
            st.info("No cases found.")
        else:
            df_cases = pd.DataFrame(cases)
            st.write("All open and reviewed cases:")
            st.dataframe(df_cases[['_id', 'status', 'risk_score', 'created_at']])
            
            case_ids = [""] + [case['_id'] for case in cases]
            selected_case_id = st.selectbox("Select a Case ID to review:", case_ids)
            
            if selected_case_id:
                selected_case = next((case for case in cases if case['_id'] == selected_case_id), None)
                
                st.subheader(f"Reviewing Case: {selected_case_id}")
                c1, c2 = st.columns(2)
                with c1:
                    st.write("**Transaction Details:**")
                    st.json(selected_case['transaction_details'])
                    st.metric("Risk Score", f"{selected_case['risk_score']:.2%}")
                with c2:
                    st.write("**Risk Factors:**")
                    display_shap_chart(selected_case['explanation'])

                st.subheader("Investigation Notes & Status Update")
                with st.form("update_case_form"):
                    current_notes = selected_case.get('notes', '')
                    notes = st.text_area("Investigator Notes", value=current_notes)
                    
                    statuses = ["Open", "In Review", "Escalated", "Closed - False Positive"]
                    current_status_index = statuses.index(selected_case['status']) if selected_case['status'] in statuses else 0
                    status = st.selectbox("Case Status", options=statuses, index=current_status_index)
                    
                    update_submitted = st.form_submit_button("Save Updates")

                    if update_submitted:
                        update_payload = {"notes": notes, "status": status}
                        try:
                            update_response = requests.put(f"{CASES_API_URL}/{selected_case_id}", json=update_payload)
                            update_response.raise_for_status()
                            st.success("Case updated successfully!")
                            # --- THIS IS THE FIX ---
                            st.rerun() # Rerun to show updated table
                        except requests.exceptions.RequestException as e:
                            st.error(f"Failed to update case: {e}")

    except requests.exceptions.RequestException as e:
        st.error(f"Failed to load cases from the API. Is the backend running? Error: {e}")