import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import tempfile
import os

# Load the trained YOLOv8 model
MODEL_PATH = "good/fresh_oranges.pt"  # Replace with your model path
model = YOLO(MODEL_PATH)

def load_image(uploaded_file):
    """Load an image from a Streamlit file uploader."""
    image = Image.open(uploaded_file).convert("RGB")  # Convert to RGB mode
    return np.array(image)

def capture_image_from_device():
    """Capture an image from a connected webcam."""
    cap = cv2.VideoCapture(0)  # 0 for default webcam
    st.info("Capturing image... Press 's' to save and exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture image.")
            break

        cv2.imshow("Webcam - Press 's' to save", frame)
        if cv2.waitKey(1) & 0xFF == ord('s'):  # Press 's' to save and exit
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            cv2.imwrite(temp_file.name, frame)
            cap.release()
            cv2.destroyAllWindows()
            return temp_file.name

    cap.release()
    cv2.destroyAllWindows()
    return None

def predict_image(image_path):
    """Run YOLO model inference and return results."""
    results = model(image_path)
    return results

def display_results(results):
    """Display YOLO model predictions."""
    for r in results:
        st.image(r.plot(), caption="Detection Results", use_column_width=True)

def main():
    st.title("Fresh Orange Quality Detector 🍊")
    st.write("Upload an image or capture one from a connected device to detect and classify oranges.")

    # Sidebar options
    option = st.sidebar.radio("Choose Input Method:", ("Upload Image", "Capture from Webcam"))

    image_path = None
    image = None

    if option == "Upload Image":
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = load_image(uploaded_file)
            image_path = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg").name
            Image.fromarray(image).save(image_path)
            st.image(image, caption="Uploaded Image", use_column_width=True)

    elif option == "Capture from Webcam":
        if st.button("Capture Image"):
            image_path = capture_image_from_device()
            if image_path:
                st.success("Image captured successfully!")
                st.image(image_path, caption="Captured Image", use_column_width=True)

    # Run model and display results
    if image_path:
        with st.spinner("Running model inference..."):
            results = predict_image(image_path)
            st.success("Inference complete!")
            display_results(results)

if __name__ == "__main__":
    main()
