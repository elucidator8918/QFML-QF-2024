import streamlit as st
import subprocess
import os
from PIL import Image

def run_simulation():
    command = [
        'python', 'dashboard_src/dashboard_utils/simulation.py', 'simulation',
        '--he',
        '--data_path', data_path,
        '--dataset', dataset,
        '--yaml_path', yaml_path,
        '--seed', str(seed),
        '--num_workers', str(num_workers),
        '--max_epochs', str(max_epochs),
        '--batch_size', str(batch_size),
        '--length', str(length),
        '--split', str(split),
        '--device', device,
        '--number_clients', str(number_clients),
        '--save_results', save_results,
        '--matrix_path', matrix_path,
        '--roc_path', roc_path,
        '--model_save', model_save,
        '--min_fit_clients', str(min_fit_clients),
        '--min_avail_clients', str(min_avail_clients),
        '--min_eval_clients', str(min_eval_clients),
        '--rounds', str(rounds),
        '--frac_fit', str(frac_fit),
        '--frac_eval', str(frac_eval),
    ]

    with st.spinner('Running simulation...'):
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        return stdout, stderr

st.set_page_config(page_title="QFed-FHE", page_icon="🚀", layout="wide")

st.title("🚀 Quantum Federated Learning with Secure Fully Homomorphic Encryption")
st.markdown("Use this app to run simulations with various configurations. Configure the parameters in the sidebar and click 'Run Simulation'.")

# Sidebar configuration
st.sidebar.header("📊 Simulation Configuration")

# Group related inputs
with st.sidebar.expander("📁 Data and Model"):
    data_path = st.text_input("Data Path", "data/")
    dataset = st.text_input("Dataset", "cifar")
    yaml_path = st.text_input("YAML Path", "./results/FL/results.yml")
    model_save = st.text_input("Model Save", "cifar_fl.pt")

with st.sidebar.expander("🔢 Training Parameters"):
    seed = st.number_input("Seed", 42)
    num_workers = st.number_input("Number of Workers", -1)
    max_epochs = st.number_input("Max Epochs", 5)
    batch_size = st.number_input("Batch Size", 256)
    length = st.number_input("Length", 256)
    split = st.number_input("Split", 10)
    device = st.text_input("Device", "cuda")

with st.sidebar.expander("🌐 Federated Learning"):
    number_clients = st.number_input("Number of Clients", 10)
    min_fit_clients = st.number_input("Min Fit Clients", 10)
    min_avail_clients = st.number_input("Min Avail Clients", 10)
    min_eval_clients = st.number_input("Min Eval Clients", 10)
    rounds = st.number_input("Rounds", 2)
    frac_fit = st.number_input("Fraction Fit", 1.0)
    frac_eval = st.number_input("Fraction Eval", 0.5)

with st.sidebar.expander("💾 Save Options"):
    save_results = st.text_input("Save Results", "results/FL/")
    matrix_path = st.text_input("Confusion Matrix Path", "confusion_matrix.png")
    roc_path = st.text_input("ROC Path", "roc.png")

if st.sidebar.button("🏃‍♂️ Run Simulation", key="run_simulation"):
    stdout, stderr = run_simulation()

    st.success("✅ Simulation finished successfully!")

    # Display plots
    col1, col2 = st.columns(2)
    
    with col1:
        if os.path.exists(matrix_path):
            st.subheader("Confusion Matrix")
            image = Image.open(matrix_path)
            st.image(image, use_column_width=True)
        else:
            st.warning("Confusion matrix not generated.")

    with col2:
        if os.path.exists(roc_path):
            st.subheader("ROC Curve")
            image = Image.open(roc_path)
            st.image(image, use_column_width=True)
        else:
            st.warning("ROC curve not generated.")

    # Execution log
    st.subheader("📜 Execution Log")
    log_expander = st.expander("View Log", expanded=False)
    with log_expander:
        st.text_area("Standard Output", stdout.decode("utf-8"), height=200)
        st.text_area("Standard Error", stderr.decode("utf-8"), height=200)