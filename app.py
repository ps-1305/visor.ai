import streamlit as st
from PIL import Image, ImageDraw
import torch
import gdown
import os
from transformers import BertTokenizer
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from streamlit_drawable_canvas import st_canvas
from model import VQAModel, preprocess_image, preprocess_question

# Configuration
st.set_page_config(
    page_title="Visor.AI",
    page_icon="favicon.ico",  # The path to your .ico file
    layout="wide"
)

@st.cache_resource  # Caches the model to avoid re-downloading/reloading each runtime
def load_model():
    # Replace with your actual file ID
    file_id = '1wVyVBB4mopJM1t9LSglr6ycnkUEpHmfm'
    url = f'https://drive.google.com/uc?id={file_id}'
    output = 'best_model.pth'

    if not os.path.exists(output):
        # Downloading the file from Google Drive
        gdown.download(url, output, quiet=False)
    
    # Load your model here
    model = VQAModel(num_classes=2)
    model.load_state_dict(torch.load(output, map_location=torch.device('cpu')))
    model.eval()
    return model

# Load model
model = load_model()

# Load and apply CSS
def load_css(css_file_path):
    with open(css_file_path) as file:
        return file.read()
css = load_css("styles.css")  # Ensure path is correct and matches the location of your styles.css
st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# Initialize session state
if 'shape_properties' not in st.session_state:
    st.session_state['shape_properties'] = [{'shape': None, 'color': None}, {'shape': None, 'color': None}]
if 'positions' not in st.session_state:
    st.session_state['positions'] = []
if 'canvas_image' not in st.session_state:
    st.session_state['canvas_image'] = None

# Helper function to draw shapes
def draw_shape(shape, color, position, img):
    draw = ImageDraw.Draw(img)
    x, y = position
    size = 40
    if shape == 'circle':
        draw.ellipse([x-size//2, y-size//2, x+size//2, y+size//2], fill=color)
    elif shape == 'square':
        draw.rectangle([x-size//2, y-size//2, x+size//2, y+size//2], fill=color)
    elif shape == 'triangle':
        points = [
            (x, y-size//2),
            (x-size//2, y+size//2),
            (x+size//2, y+size//2)
        ]
        draw.polygon(points, fill=color)

# Sidebar navigation
st.sidebar.title("Navigation")
if st.sidebar.button('Demo'):
    st.session_state['page'] = 'Demo'
if st.sidebar.button('Code'):
    st.session_state['page'] = 'Code'
if st.sidebar.button('Results'):
    st.session_state['page'] = 'Results'

page = st.session_state.get('page', 'Demo')

if page == 'Demo':
    st.title("Demo")
    st.write("Select properties for two shapes and place them by selecting points on the canvas.")

    # Define available shapes and colors
    shapes = ['circle', 'square', 'triangle']
    colors = ['red', 'green', 'blue']
    color_values = {
        'red': (255, 0, 0),
        'green': (0, 255, 0),
        'blue': (0, 0, 255),
    }

    # Column layout for shape properties
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Properties (1)")
        st.session_state['shape_properties'][0]['shape'] = st.selectbox("Select Shape for Shape 1", shapes, key="shape1_select")
        st.session_state['shape_properties'][0]['color'] = st.selectbox("Select Color for Shape 1", colors, key="color1_select")
        confirm_shape_properties = st.button("Confirm shape properties")
        st.header("Question")
        
    with col2:
        st.subheader("Properties (2)")
        st.session_state['shape_properties'][1]['shape'] = st.selectbox("Select Shape for Shape 2", shapes, key="shape2_select")
        st.session_state['shape_properties'][1]['color'] = st.selectbox("Select Color for Shape 2", colors, key="color2_select")

    with col3:
        st.subheader("Positions")
        canvas_result = st_canvas(
            fill_color="rgba(0, 0, 0, 0)",
            stroke_width=1,
            stroke_color="rgba(0, 0, 0, 1)",
            background_color="rgba(255, 255, 255, 1)",
            width=224,
            height=224,
            drawing_mode="point",
            display_toolbar=False,
            key="canvas"
        )
        pos_confirm = st.button("Confirm positions")

    # Display shape properties confirmation
    if confirm_shape_properties:
        st.success("Shape properties confirmed!")

    # Capture positions
    if canvas_result.json_data is not None:
        points = canvas_result.json_data["objects"]
        if len(points) <= 2:
            st.session_state['positions'] = [(point['left'], point['top']) for point in points]

    # Button to confirm positions and draw the shapes
    if len(st.session_state['positions']) == 2 and pos_confirm:
        st.success("Positions confirmed!")
        # Draw shapes based on selected properties and positions
        img = Image.new('RGB', (224, 224), 'white')
        for i, position in enumerate(st.session_state['positions']):
            shape_data = st.session_state['shape_properties'][i]
            draw_shape(shape_data['shape'], color_values[shape_data['color']], position, img)

        st.session_state['canvas_image'] = img  # Store the image in session state

        scaled_img = img.resize((224*3, 224*3), Image.NEAREST)
        st.image(scaled_img, caption="Input", use_container_width=False)

    # Form the question based on user input
    data = st.session_state['shape_properties']
    datalist = [f"{data[0]['color']} {data[0]['shape']}", f"{data[1]['color']} {data[1]['shape']}"]
    col1, col2, col3, col4, col5 = st.columns([0.35,1,1,0.1,1], gap="small", vertical_alignment="center")
    with col1:
        st.text("Is there a")

    with col2:
        selection_1 = st.selectbox("Select shape", datalist)
    
    with col3:
        pos = st.selectbox("Select position", ["above", "below", "right of", "left of"], key="one")

    with col4:
        st.text("a")
    
    with col5:
        selection_2 = st.selectbox("Select shape", list(set(datalist)-set(selection_1)), key="two")

    submit = st.button("Submit", key="submit-button")
    
    col1, col2, col3 = st.columns([1,3,1])
    if submit and len(st.session_state['positions']) == 2 and st.session_state['canvas_image'] is not None:
        with col2:
            subcol1, subcol2 = st.columns(2)
            question = f"Is there a {selection_1} {pos} a {selection_2}?"

            # Preprocess the inputs
            with subcol1:
                st.image(st.session_state['canvas_image'], caption="Input")
            image_tensor = preprocess_image()(st.session_state['canvas_image']).unsqueeze(0)
            tokens = preprocess_question(question)
            input_ids = tokens['input_ids']
            attention_mask = tokens['attention_mask']

            # Make prediction
            with torch.no_grad():
                logits = model(image_tensor, input_ids, attention_mask)
                prediction = torch.argmax(logits, dim=1).item()

            answer = 'YES' if prediction == 0 else 'NO'
            with subcol2:
                st.text(f"Your question: {question}")
                st.success(f"Model Prediction: {answer}")

elif page == 'Code':
    st.title("Code Page")
    st.write("Here is the code section.")

elif page == 'Results':
    st.title("Results Page")
