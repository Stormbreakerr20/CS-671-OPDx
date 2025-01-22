import streamlit as st
import streamlit.components.v1 as com
import subprocess
import aiohttp
import asyncio
import os

st.set_option('deprecation.showfileUploaderEncoding', False)
st.set_page_config(page_title="OPDx", page_icon="🩺", layout="wide")

# Get the parent directory
parent_directory = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
new_parent_directory = os.path.join(os.path.dirname(parent_directory), "app")

# Streamed response emulator
async def response_generator(prompt):
    async with aiohttp.ClientSession() as session:
        async with session.post(f'http://127.0.0.1:8000/generate?inp={prompt}') as response:
            text = await response.json()
            print(text)
            return text['message']
        
async def uploadImage(file):
    async with aiohttp.ClientSession() as session:
        async with session.post(f'http://127.0.0.1:8000/upload_img?inp={file}') as response:
            text = await response.json()
            if(text['message'] == "200"):
                return "Image uploaded successfully"
            else: 
                return "Error in uploading image"
            
async def uploadPdf():
    async with aiohttp.ClientSession() as session:
        async with session.post('http://127.0.0.1:8000/upload') as response:
            text = await response.json()
            if(text['message'] == "200"):
                return "PDF uploaded successfully"
            else: 
                return "Error in uploading file"

def colored_markdown(text: str, color: str):
    return f'<p class="title" style="background-color: #fff; color: {color}; padding: 5px; line-height: 1">{text}</p>'

sidebar = st.sidebar
col1, col2 = st.columns([1, 3])

# Display Lottie animation in the first column
with col1:
    com.iframe("https://lottie.host/embed/5899ceed-3498-4546-8ebf-b25561f40002/Xnif8r8nZ4.json", height=400, width=950)

try:
    st.markdown(colored_markdown(f"Luv", "#007bff"),
                unsafe_allow_html=True)  # Blue color
    st.markdown(colored_markdown("How can I help you today ?", "#39A5A9"),
                unsafe_allow_html=True)  # Red color
except:

    st.markdown(colored_markdown(f"Luv", "#007bff"),
                unsafe_allow_html=True)  # Blue color
    st.markdown(colored_markdown("How can I help you today ?", "#39A5A9"),
                unsafe_allow_html=True)  # Red color

st.markdown(
    """
    <style>
        .title {
            text-align: left; 
            font-size: 60px;
            margin-bottom: 0px;
            padding-bottom: 5px;
            display: inline-block;  
        }
        .subtitle {
            text-align: left;
            font-size: 24px;
            color: #333333; 
            margin-top: 5px; 
        }
        .square-container {
            display: flex;
            flex-wrap: wrap;
        }
        .square {
            width: 150px;
            height: 150px;
            background-color: #36A5A9;
            margin: 10px;
            margin-top: 30px;  
            margin-bottom: 50px;  
            color: #ffffff;
            padding: 10px;
            text-align: left;
            font-size: 14px;
            line-height: 1.5;
            border-radius: 16px;
            position: relative;  /* Enable relative positioning for image */
        }
        .square-image {
            position: absolute;  /* Make image absolute within square */
            bottom: 5px;  /* Position image at bottom */
            right: 5px;  /* Position image at right */
            width: 20px;
            height: 20px;
        }
        .input-container {
            display: flex;
            align-items: center;
            position: relative;
            margin-top: 20px;
        }
        .input-text {
            flex: 1;
            height: 40px;
            padding: 10px;
            font-size: 16px;
            border-radius: 12px;
            border: 1px solid #ccc;
            margin-right: 10px;
        }
        .button-container {
            display: flex;
            gap: 0px;
        }
        .button {
            
            height: 40px;
            width: 40px;
            margin: 0px;
            padding: 0px;
            display: flex;
            justify-content: center;
            align-items: center;
            # background-color: #39A5A9;
            color: #fff;
            border-radius: 8px;
            cursor: pointer;
        }
        .button svg {
            width: 24px;
            height: 24px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown(f'<img src="https://i.imgur.com/ngr2HSn.png" width="200">',
                    unsafe_allow_html=True)  # Load image from Imgur with Bitly link

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        generated = asyncio.run(response_generator(prompt))
        response = st.write(generated)
    st.session_state.messages.append({"role": "assistant", "content": generated})

# File uploader
uploaded_file = st.file_uploader(
    "Upload a pdf or image file",
    type=["pdf", "jpg", "png", "jpeg"],
    help="Upload a file to ask related questions."
)

# Check if a file is uploaded
if uploaded_file is not None:
    print("File uploaded")  # Debugging statement
    
    # Check the type of the uploaded file
    if uploaded_file.type == "application/pdf":
        file_extension = "pdf"
    else:
        file_extension = uploaded_file.name.split(".")[-1]
    
    # Save the file in the parent directory
    with open(os.path.join(new_parent_directory, f"uploaded_file.{file_extension}"), "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Display a success message 
    if uploaded_file.type == "application/pdf":
        pdf_response = asyncio.run(uploadPdf())
        subprocess.Popen("rm ./../uploaded_file*",shell=True)
    else:
        img_response = asyncio.run(uploadImage(f"uploaded_file.{file_extension}"))
        
    st.write(f"File uploaded successfully: {uploaded_file.name}")
    print("Success from backend")

st.sidebar.write("##")
st.sidebar.markdown("<h2 style='text-align: center;'>User Dashboard</h2>", unsafe_allow_html=True)
st.sidebar.write("##")