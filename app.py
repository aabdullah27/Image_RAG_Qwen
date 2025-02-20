import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from qdrant_client import QdrantClient, models
import fitz  # PyMuPDF for PDF processing
import io
import tempfile
import os
from PIL import Image

# Page config
st.set_page_config(
    page_title="Multimodal RAG powered by Qwen2.5-VL",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'ready_to_chat' not in st.session_state:
    st.session_state.ready_to_chat = False
if 'collection_name' not in st.session_state:
    st.session_state.collection_name = "multimodal-rag"
if 'processed_pages' not in st.session_state:
    st.session_state.processed_pages = []

def process_pdf_to_images(pdf_file):
    """Convert PDF pages to images."""
    images = []
    
    # Create a temporary file to save the PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(pdf_file.getvalue())
        tmp_file.flush()
        
        # Open the PDF with PyMuPDF
        doc = fitz.open(tmp_file.name)
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            pix = page.get_pixmap()
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            images.append(img)
            
        doc.close()
    
    # Clean up temporary file
    os.unlink(tmp_file.name)
    return images

# Initialize QDrant client
@st.cache_resource
def init_qdrant():
    return QdrantClient(url="http://localhost:6333")

# Initialize model and processor
@st.cache_resource
def init_model():
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    return model, processor

# Sidebar for document upload
with st.sidebar:
    st.title("Add your documents!")
    uploaded_file = st.file_uploader("Choose your .pdf file", type=['pdf'])
    
    if uploaded_file:
        st.write(f"{uploaded_file.name}")
        st.write(f"{uploaded_file.size} bytes")
        
        # Clear button
        if st.button("Clear"):
            st.session_state.ready_to_chat = False
            st.session_state.processed_pages = []
            st.rerun()

# Main chat interface
st.title("Multimodal RAG powered by Qwen2.5-VL")

if uploaded_file and not st.session_state.ready_to_chat:
    with st.spinner("Indexing your document..."):
        try:
            # Process document
            images = process_pdf_to_images(uploaded_file)
            st.session_state.processed_pages = images
            st.session_state.ready_to_chat = True
            
            # Initialize QDrant collection
            client = init_qdrant()
            if not client.collection_exists(st.session_state.collection_name):
                client.create_collection(
                    collection_name=st.session_state.collection_name,
                    vectors_config=models.VectorParams(
                        size=1024,  # Adjust based on your embedding size
                        distance=models.Distance.COSINE
                    )
                )
            
            st.success("Ready to Chat!")
            
        except Exception as e:
            st.error(f"Error processing document: {str(e)}")

# Chat interface
if st.session_state.ready_to_chat:
    # Display PDF preview
    st.sidebar.subheader("PDF Preview")
    if st.session_state.processed_pages:
        st.sidebar.image(st.session_state.processed_pages[0], use_column_width=True)
    
    # Chat interface
    query = st.text_input("What's up?")
    
    if query:
        try:
            model, processor = init_model()
            
            # Prepare the conversation
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": query},
                        {"type": "image", "image": st.session_state.processed_pages[0]}
                    ]
                }
            ]
            
            # Generate response
            inputs = processor.apply_chat_template(messages, tokenize=True, return_tensors="pt")
            inputs = inputs.to(model.device)
            
            with torch.no_grad():
                output_ids = model.generate(
                    inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.8
                )
            
            response = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
            st.write(response)
            
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")

else:
    st.info("Please upload a document to start chatting!")

# Add footer
st.markdown("---")
st.markdown("Powered by Qwen2.5-VL and Streamlit")
