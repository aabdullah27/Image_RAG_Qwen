import streamlit as st
import torch
from transformers import QwenForConditionalGeneration, AutoProcessor
from colpali_engine import ColPali, ColPaliProcessor
from qdrant_client import QdrantClient, models
from PIL import Image
import fitz  # PyMuPDF
import io
import tempfile
import os

# Page config
st.set_page_config(page_title="Multimodal RAG powered by Qwen2.5-VL", layout="wide")

# Initialize session state
if 'ready_to_chat' not in st.session_state:
    st.session_state.ready_to_chat = False
if 'client' not in st.session_state:
    st.session_state.client = None
if 'collection_name' not in st.session_state:
    st.session_state.collection_name = "qwen-colpali-multimodalRAG"

# Sidebar for uploading documents
with st.sidebar:
    st.title("Add your documents!")
    uploaded_file = st.file_uploader("Drag and drop file here", type=['pdf'])
    
    if uploaded_file:
        st.write(f"{uploaded_file.name}\n{uploaded_file.size} bytes")
        
        # Clear button
        if st.button("Clear"):
            uploaded_file = None
            st.session_state.ready_to_chat = False
            st.experimental_rerun()

# Initialize QDrant client
@st.cache_resource
def init_qdrant():
    return QdrantClient(url="http://localhost:6333")

# Initialize Qwen model and processor
@st.cache_resource
def init_qwen():
    model_path = "Qwen/Qwen2.5-VL-3B-Instruct"
    model = QwenForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
        cache_dir="/teamspace/studios/this_studio/Qwen/hf_cache"
    )
    processor = AutoProcessor.from_pretrained(
        model_path,
        cache_dir="/teamspace/studios/this_studio/Qwen/hf_cache"
    )
    return model, processor

# Initialize ColPali model and processor
@st.cache_resource
def init_colpali():
    model_name = "vidore/colpali-v1.2"
    embed_model = ColPali.from_pretrained(model_name)
    processor = ColPaliProcessor.from_pretrained(model_name)
    return embed_model, processor

# Function to process PDF and convert pages to images
def process_pdf_to_images(pdf_file):
    images = []
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(pdf_file.getvalue())
        tmp_file.flush()
        doc = fitz.open(tmp_file.name)
        for page in doc:
            pix = page.get_pixmap()
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            images.append(img)
        doc.close()
    os.unlink(tmp_file.name)
    return images

# Process document and create embeddings
def process_document(file, embed_model, processor, client):
    # Create collection if it doesn't exist
    if not client.collection_exists(st.session_state.collection_name):
        client.create_collection(
            collection_name=st.session_state.collection_name,
            vectors_config=models.VectorParams(
                size=128,
                distance=models.Distance.COSINE
            )
        )
    
    # Process document pages
    images = process_pdf_to_images(file)  # Make sure this function works as intended
    embeddings = []
    
    for i, image in enumerate(images):
        # Process image with ColPali
        processed_image = processor(images=image, return_tensors="pt")
        embedding = embed_model.get_image_features(**processed_image)
        
        # Store in QDrant
        point = models.PointStruct(
            id=i,
            vector=embedding.detach().cpu().numpy().flatten(),
            payload={"page_number": i}
        )
        client.upsert(
            collection_name=st.session_state.collection_name,
            points=[point]
        )
        embeddings.append(embedding)
    
    return embeddings

# Main chat interface
st.title("Multimodal RAG powered by Qwen2.5-VL")

if uploaded_file:
    if not st.session_state.ready_to_chat:
        with st.spinner("Indexing your document..."):
            try:
                client = init_qdrant()
                embed_model, colpali_processor = init_colpali()
                qwen_model, qwen_processor = init_qwen()
                
                process_document(uploaded_file, embed_model, colpali_processor, client)
                st.session_state.client = client
                st.session_state.ready_to_chat = True
                
                st.success("Ready to Chat!")
            except Exception as e:
                st.exception(f"Error during document processing: {e}")

if st.session_state.ready_to_chat:
    query = st.text_input("Ask a question about your document:")
    
    if query:
        try:
            # Generate query embedding
            processed_query = colpali_processor(text=query, return_tensors="pt")
            query_embedding = embed_model.get_text_features(**processed_query)
            
            # Retrieve relevant content
            search_result = st.session_state.client.search(
                collection_name=st.session_state.collection_name,
                query_vector=query_embedding.detach().cpu().numpy()
            )
            
            # Construct conversation message (adjust key names based on your Qdrant response)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": search_result['result'][0]['payload'].get('image', None)},
                        {"type": "text", "text": f"Answer the user query: {query}, based on the image provided."}
                    ]
                }
            ]
            
            inputs = qwen_processor(messages, return_tensors="pt")
            output = qwen_model.generate(**inputs, max_new_tokens=128)
            response = qwen_processor.decode(output[0], skip_special_tokens=True)
            
            st.write(response)
        except Exception as e:
            st.exception(f"Error during query processing: {e}")
else:
    st.info("Please upload a document to start chatting!")
