import streamlit as st
import cv2
from ultralytics import YOLO
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load YOLOv8 Models
@st.cache_resource
def load_model(model_path):
    try:
        model = YOLO(model_path)
        st.success(f"Model loaded: {model_path}")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Tier system for confidence levels
def classify_tier(conf):
    if conf >= 0.9:
        return 'S'  # Export quality
    elif conf >= 0.75:
        return 'A'  # Supermarkets
    elif conf >= 0.6:
        return 'B'  # Local Markets
    elif conf >= 0.4:
        return 'C'  # Farmers Markets
    else:
        return 'F'  # Recycling

# Cost per tier
TIER_COST = {'S': 37, 'A': 32, 'B': 26, 'C': 25, 'F': 15}  # Prices in BDT

# Detection function with tier classification
def run_detection_with_tiers(image, model, conf_threshold, label_rename=None):
    if model is None:
        st.error("Model not loaded.")
        return [], np.array(image)

    image = image.convert("RGB")
    frame = np.array(image)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    results = model(frame, conf=conf_threshold)
    detections = []

    for result in results:
        for box in result.boxes.data.clone():
            conf = float(box[4])
            label = int(box[-1])
            tier = classify_tier(conf)

            if label_rename and label == 1:  # Relabel "saine" to "orange_bad"
                label = 0
                box[-1] = label

            detections.append({
                'label': model.names[label] if not label_rename else "orange_bad",
                'confidence': conf,
                'tier': tier
            })

    annotated_frame = results[0].plot()
    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    return detections, annotated_frame

# Streamlit UI
st.title("IoT-Enabled Machine Learning Solution for Automated Quality Evaluation of Orange Fruit Through Image Processing 🍊")
st.write("Capture an image from your webcam or upload an image to detect fresh and bad oranges with tier classification and cost analysis.")

# Load models
fresh_model = load_model("good/fresh_oranges.pt")
bad_model = load_model("bad/bad_oranges.pt")

# Confidence threshold slider
conf_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.25, 0.05)

# Webcam image capture or upload
option = st.radio("Select Input Method", ("Upload Image", "Capture Image from Webcam"))

if option == "Capture Image from Webcam":
    captured_image = st.camera_input("Capture Image")  # Using Streamlit's camera input component
    if captured_image:
        image = Image.open(captured_image)
        st.image(image, caption="Captured Image", use_container_width=True)
else:
    uploaded_file = st.file_uploader("Choose an image to upload...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)

if 'image' in locals():
    if st.button("Run Detection"):
        # Run detection on both models
        fresh_detections, fresh_img = run_detection_with_tiers(image, fresh_model, conf_threshold)
        bad_detections, bad_img = run_detection_with_tiers(image, bad_model, conf_threshold, label_rename=True)

        # Display results
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Fresh Orange Model Results")
            st.image(fresh_img, use_container_width=True, caption="Fresh Oranges")
        with col2:
            st.subheader("Bad Orange Model Results")
            st.image(bad_img, use_container_width=True, caption="Bad Oranges (orange_bad)")

        # Combine detections and split into separate dataframes
        df_fresh = pd.DataFrame(fresh_detections)
        df_bad = pd.DataFrame(bad_detections)

        if not df_fresh.empty or not df_bad.empty:
            st.subheader("Tier Classification and Cost Analysis")

            # Process fresh oranges
            if not df_fresh.empty:
                st.write("### Fresh Oranges")
                fresh_summary = df_fresh['tier'].value_counts().reset_index()
                fresh_summary.columns = ['Tier', 'Count']
                fresh_summary['Cost per Unit (BDT)'] = fresh_summary['Tier'].map(TIER_COST)
                fresh_summary['Total Cost (BDT)'] = fresh_summary['Count'] * fresh_summary['Cost per Unit (BDT)']
                st.table(fresh_summary)

                # Fresh oranges graph
                st.write("#### Tier Distribution (Fresh Oranges)")
                fig1, ax1 = plt.subplots()
                ax1.bar(fresh_summary['Tier'], fresh_summary['Count'], color='green')
                ax1.set_xlabel("Tier")
                ax1.set_ylabel("Count")
                ax1.set_title("Fresh Oranges Tier Distribution")
                st.pyplot(fig1)

            # Process bad oranges
            if not df_bad.empty:
                st.write("### Bad Oranges")
                bad_summary = df_bad['tier'].value_counts().reset_index()
                bad_summary.columns = ['Tier', 'Count']
                bad_summary['Cost per Unit (BDT)'] = bad_summary['Tier'].map(TIER_COST)
                bad_summary['Total Cost (BDT)'] = bad_summary['Count'] * bad_summary['Cost per Unit (BDT)']
                st.table(bad_summary)

                # Bad oranges graph
                st.write("#### Tier Distribution (Bad Oranges)")
                fig2, ax2 = plt.subplots()
                ax2.bar(bad_summary['Tier'], bad_summary['Count'], color='red')
                ax2.set_xlabel("Tier")
                ax2.set_ylabel("Count")
                ax2.set_title("Bad Oranges Tier Distribution")
                st.pyplot(fig2)

            # Combined Stock Distribution
            st.write("### Combined Stock Distribution")
            combined_df = pd.concat([df_fresh, df_bad], axis=0)
            combined_summary = combined_df['tier'].value_counts().reset_index()
            combined_summary.columns = ['Tier', 'Count']
            fig3, ax3 = plt.subplots()
            ax3.pie(combined_summary['Count'], labels=combined_summary['Tier'], autopct='%1.1f%%', startangle=90)
            ax3.axis('equal')
            st.pyplot(fig3)

            # Total stock
            total_stock = combined_summary['Count'].sum()
            st.write(f"### Total Stock: {total_stock} oranges")

            # Example Distributors
            st.subheader("Distribution Recommendations (Bangladesh Context)")
            st.write("""
            - **S Tier** (Export Quality): Countries like UAE, UK, Germany
            - **A Tier** (Supermarkets): Shwapno, Meena Bazar, Agora
            - **B & C Tier** (Local Markets): Local fruit vendors, Farmers Markets
            - **F Tier** (Recycling): Processing plants for juice and compost
            """)
        else:
            st.warning("No detections found in the image!")
