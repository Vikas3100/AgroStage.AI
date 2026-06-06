import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
from utils.prediction import predict_cnn, predict_resnet, predict_yolo

# Page Config
st.set_page_config(
    page_title="Crop Growth Stage Analyzer",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .hero-banner {
        background: linear-gradient(135deg, #134e5e 0%, #71b280 100%);
        padding: 3rem 2rem;
        border-radius: 16px;
        color: white;
        text-align: center;
        margin-bottom: 2.5rem;
        box-shadow: 0 10px 25px rgba(19, 78, 94, 0.15);
    }
    .hero-banner h1 {
        font-weight: 700;
        margin-bottom: 0.5rem;
        color: white !important;
        font-size: 2.6rem;
    }
    .hero-banner p {
        font-size: 1.1rem;
        opacity: 0.95;
        font-weight: 400;
        max-width: 800px;
        margin: 0 auto;
    }
    
    .section-title {
        font-weight: 600;
        font-size: 1.4rem;
        margin-bottom: 1.2rem;
        color: #1e293b;
        border-left: 4px solid #71b280;
        padding-left: 0.6rem;
        margin-top: 1rem;
    }
    
    .card {
        background: #ffffff;
        border-radius: 16px;
        padding: 2rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
    }
    
    .result-badge {
        display: inline-block;
        padding: 0.35rem 0.8rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 1rem;
    }
    .badge-yolo { background-color: #ffe4e6; color: #e11d48; }
    .badge-cnn { background-color: #ecfdf5; color: #059669; }
    .badge-resnet { background-color: #eff6ff; color: #2563eb; }
    
    .stage-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.2rem;
    }
    .stage-subtitle {
        font-size: 1rem;
        color: #64748b;
        font-weight: 500;
        margin-bottom: 1.5rem;
    }
    
    .metric-value {
        font-size: 2.8rem;
        font-weight: 800;
        color: #10b981;
        line-height: 1;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.25rem;
    }
    
    .custom-progress {
        width: 100%;
        background-color: #f1f5f9;
        border-radius: 9999px;
        height: 12px;
        overflow: hidden;
        margin: 1.2rem 0;
    }
    .custom-progress-fill {
        height: 100%;
        background: linear-gradient(90deg, #10b981 0%, #34d399 100%);
        border-radius: 9999px;
    }
    
    .description-text {
        font-size: 0.95rem;
        color: #475569;
        line-height: 1.6;
        border-top: 1px solid #e2e8f0;
        padding-top: 1rem;
        margin-top: 1rem;
    }
    
    .sidebar-desc {
        font-size: 0.9rem;
        color: #475569;
        line-height: 1.5;
        margin-bottom: 1.5rem;
    }
    
    /* Center Streamlit buttons and give clean hover behaviors */
    div.stButton > button:first-child {
        width: 100%;
        background-color: #ffffff;
        color: #0f172a;
        border: 1px solid #cbd5e1;
        transition: all 0.2s ease;
        font-weight: 500;
        border-radius: 8px;
    }
    div.stButton > button:first-child:hover {
        background-color: #f8fafc;
        border-color: #71b280;
        color: #71b280;
        box-shadow: 0 4px 12px rgba(113, 178, 128, 0.12);
        transform: translateY(-1px);
    }
    div.stButton > button:first-child:active {
        transform: translateY(0px);
    }
    
    /* Primary button custom styles */
    div.stButton > button[kind="primary"] {
        width: 100%;
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%) !important;
        color: white !important;
        border: none !important;
        font-weight: 600 !important;
        padding: 0.6rem 1rem !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 15px rgba(17, 153, 142, 0.25) !important;
        transition: all 0.3s ease !important;
    }
    div.stButton > button[kind="primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(17, 153, 142, 0.35) !important;
    }
    div.stButton > button[kind="primary"]:active {
        transform: translateY(0px) !important;
    }
</style>
""", unsafe_allow_html=True)

# Custom Title Banner
st.markdown("""
<div class="hero-banner">
    <h1>🌱 Crop Growth Stage Analyzer</h1>
    <p>Upload a crop image or select an example to identify its precise growth phase using deep learning models.</p>
</div>
""", unsafe_allow_html=True)

# State initialization
if 'active_image_source' not in st.session_state:
    st.session_state['active_image_source'] = None
if 'example_path' not in st.session_state:
    st.session_state['example_path'] = None
if 'uploaded_file' not in st.session_state:
    st.session_state['uploaded_file'] = None
if 'uploader_idx' not in st.session_state:
    st.session_state['uploader_idx'] = 0
if 'run_analysis' not in st.session_state:
    st.session_state['run_analysis'] = False
if 'last_image_id' not in st.session_state:
    st.session_state['last_image_id'] = None
if 'last_model_choice' not in st.session_state:
    st.session_state['last_model_choice'] = None

# Callback for example selection
def select_example(path):
    st.session_state['active_image_source'] = 'example'
    st.session_state['example_path'] = path
    # Increment key index to clear/reset the file uploader widget
    st.session_state['uploader_idx'] += 1
    st.session_state['uploaded_file'] = None
    st.session_state['run_analysis'] = False

# Sidebar Configuration
st.sidebar.markdown("### ⚙️ Control Panel")

model_choice = st.sidebar.selectbox(
    "Select Model",
    ["Custom CNN", "ResNet50", "YOLOv8"],
    key="model_select"
)

# Model description mapping
model_descriptions = {
    "Custom CNN": "A custom 3-layer convolutional network trained specifically for fast, CPU-friendly classification of agricultural growth stages.",
    "ResNet50": "An industry-standard deep residual network loaded with custom transfer-learning weights to extract rich visual features.",
    "YOLOv8": "A single-stage object detector running classification based on localized regional characteristics."
}

st.sidebar.markdown(f"""
**Selected Model Info:**
<div class="sidebar-desc">
    {model_descriptions[model_choice]}
</div>
""", unsafe_allow_html=True)

# Reset prediction if model choice changes
if model_choice != st.session_state['last_model_choice']:
    st.session_state['run_analysis'] = False
    st.session_state['last_model_choice'] = model_choice

# Main Section - Grid Layout
col_left, col_right = st.columns([7, 5], gap="large")

with col_left:
    st.markdown('<div class="section-title">Upload Image</div>', unsafe_allow_html=True)
    
    # 1. ALWAYS ON TOP: Upload Button
    uploader_key = f"custom_uploader_{st.session_state['uploader_idx']}"
    uploaded_file = st.file_uploader(
        "Choose a crop image...",
        type=["jpg", "png", "jpeg"],
        key=uploader_key
    )
    
    if uploaded_file is not None:
        st.session_state['active_image_source'] = 'upload'
        st.session_state['uploaded_file'] = uploaded_file
        st.session_state['example_path'] = None
    
    # Resolve the active image ID
    active_image_id = None
    image_to_process = None
    display_img = None
    
    if st.session_state['active_image_source'] == 'upload' and st.session_state['uploaded_file']:
        uploaded_file = st.session_state['uploaded_file']
        active_image_id = f"upload_{uploaded_file.name}_{uploaded_file.size}"
        
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image_to_process = cv2.imdecode(file_bytes, 1)
        uploaded_file.seek(0)
        display_img = Image.open(uploaded_file)
        
    elif st.session_state['active_image_source'] == 'example' and st.session_state['example_path']:
        image_path = st.session_state['example_path']
        active_image_id = f"example_{image_path}"
        
        image_to_process = cv2.imread(image_path)
        display_img = Image.open(image_path)

    # Reset analysis if image has changed
    if active_image_id != st.session_state['last_image_id']:
        st.session_state['run_analysis'] = False
        st.session_state['last_image_id'] = active_image_id

    # 2. SHOW EXAMPLES ONLY IF NO IMAGE IS UPLOADED
    if st.session_state['active_image_source'] != 'upload':
        st.markdown('<div class="section-title">Or Select an Example Image</div>', unsafe_allow_html=True)
        
        examples_dir = "examples"
        example_images = [
            ("Example 1", "germination.jpg"),
            ("Example 2", "vegetative.jpg"),
            ("Example 3", "flowering.jpg"),
            ("Example 4", "harvesting.jpg")
        ]
        
        # Display examples as structured grid
        cols = st.columns(4)
        for idx, (label, fname) in enumerate(example_images):
            path = os.path.join(examples_dir, fname)
            with cols[idx]:
                if os.path.exists(path):
                    st.image(path, caption=label, use_container_width=True)
                    st.button(f"Analyze", key=f"btn_{fname}", on_click=select_example, args=(path,))
                else:
                    st.error(f"Missing {fname}")

    # Display Preview and Submit Button
    if image_to_process is not None:
        st.markdown('<div class="section-title">Selected Image Preview</div>', unsafe_allow_html=True)
        st.image(display_img, use_container_width=True)
        
        # Centered Run Classification submit button
        if st.button("🔍 Run Classification", type="primary", key="submit_btn"):
            st.session_state['run_analysis'] = True
            st.rerun()
    else:
        st.info("👈 Please select an example crop image or upload a custom one to begin analysis.")

with col_right:
    st.markdown('<div class="section-title">Classification Result</div>', unsafe_allow_html=True)
    
    if image_to_process is not None:
        if st.session_state['run_analysis']:
            with st.spinner("Analyzing image..."):
                # Execute prediction based on selected model
                if model_choice == "Custom CNN":
                    pred_class, confidence = predict_cnn(image_to_process)
                    badge_class = "badge-cnn"
                elif model_choice == "ResNet50":
                    pred_class, confidence = predict_resnet(image_to_process)
                    badge_class = "badge-resnet"
                else:
                    pred_class, confidence = predict_yolo(image_to_process)
                    badge_class = "badge-yolo"
                    
                conf_percent = confidence * 100
                
                # Growth stage detailed descriptions
                stage_info = {
                    "Germination": "The initial growth stage where the seed sprouts, establishing root networks and the first tiny leaves emerge above the soil.",
                    "Vegetative": "A rapid growth phase focused on stems, leaves, and overall foliage development. The plant is actively accumulating energy for production.",
                    "Flowering": "The reproductive stage where flowers develop. This is a crucial phase determining final yields, requiring optimal nutrition and hydration.",
                    "Harvesting": "The crop has reached mature stage and is ready for gathering. Color changes and drying indicate readiness for collection."
                }
                
                description = stage_info.get(pred_class, "Growth stage detected successfully.")
                
                # HTML for styled result card (Written with zero indentation to prevent raw text rendering)
                html_code = f"""<div class="card">
<span class="result-badge {badge_class}">{model_choice} Prediction</span>
<div class="stage-title">{pred_class}</div>
<div class="stage-subtitle">Crop Growth Stage Category</div>
<div style="margin-top: 1.5rem;">
<div class="metric-value">{conf_percent:.1f}%</div>
<div class="metric-label">Confidence Score</div>
</div>
<div class="custom-progress">
<div class="custom-progress-fill" style="width: {conf_percent}%;"></div>
</div>
<div class="description-text">
<strong>Stage Summary:</strong><br/>
{description}
</div>
</div>"""
                
                st.markdown(html_code, unsafe_allow_html=True)
                
                # Add extra agricultural guide
                st.info(f"💡 Recommended action for **{pred_class}** stage: Monitor environmental conditions and customize fertilization/watering patterns accordingly.")
        else:
            st.markdown("""
            <div class="card" style="text-align: center; color: #64748b;">
                <h3 style="color: #475569; margin-bottom: 0.5rem;">Ready for Inference</h3>
                <p>Click the <b>Run Classification</b> button below the image preview to start the deep learning analysis.</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("No image selected for classification. Choose an image on the left to see results.")
