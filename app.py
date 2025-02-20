import streamlit as st
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qdrant_client import QdrantClient, models
from colpali_engine.models import ColPali, ColPaliProcessor
import os

# Page config
st.set_page_config(page_title="Multimodal RAG powered by Qwen2.5-VL", layout="wide")

# Initialize session state
if 'ready_to_chat' not in st.session_state:
    st.session_state.ready_to_chat = False
if 'client' not in st.session_state:
    st.session_state.client = None
if 'collection_name' not in st.session_state:
    st.session_state.collection_name = "gwen-colpali-multimodalRAG"

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
            st.rerun()

# Initialize QDrant client
@st.cache_resource
def init_qdrant():
    return QdrantClient(url="http://localhost:6333")

# Initialize Qwen model and processor
@st.cache_resource
def init_qwen():
    model_path = "Qwen/Qwen2.5-VL-3B-Instruct"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
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

# Initialize ColPali
@st.cache_resource
def init_colpali():
    model_name = "vidore/colpali-v1.2"
    embed_model = ColPali.from_pretrained(model_name)
    processor = ColPaliProcessor.from_pretrained(model_name)
    return embed_model, processor

# Process document and create embeddings
def process_document(file, embed_model, processor, client):
    # Create collection if it doesn't exist
    if not client.collection_exists(st.session_state.collection_name):
        client.create_collection(
            collection_name=st.session_state.collection_name,
            on_disk_payload=True,
            vectors_config=models.VectorParams(
                size=128,
                distance=models.Distance.COSINE,
                on_disk=True,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM
                )
            )
        )
    
    # Process document pages
    images = process_pdf_to_images(file)  # You'll need to implement this
    embeddings = []
    
    for i, image in enumerate(images):
        # Process image with ColPali
        processed_image = processor.process_images(image)
        embedding = embed_model(processed_image)
        
        # Store in QDrant
        point = models.PointStruct(
            id=i,
            vector=embedding,
            payload={"image": image}  # You might want to store the image path instead
        )
        client.upsert(
            collection_name=st.session_state.collection_name,
            points=point
        )
        embeddings.append(embedding)

    return embeddings

# Main chat interface
st.title("Multimodal RAG powered by Qwen2.5-VL")

if uploaded_file:
    if not st.session_state.ready_to_chat:
        with st.spinner("Indexing your document..."):
            # Initialize all components
            client = init_qdrant()
            embed_model, colpali_processor = init_colpali()
            qwen_model, qwen_processor = init_qwen()
            
            # Process document
            embeddings = process_document(uploaded_file, embed_model, colpali_processor, client)
            st.session_state.client = client
            st.session_state.ready_to_chat = True
            
        st.success("Ready to Chat!")

# Chat interface
if st.session_state.ready_to_chat:
    query = st.text_input("Ask a question about your document:")
    
    if query:
        # Generate query embedding
        query_embedding = embed_model(query)
        
        # Retrieve relevant content
        result = st.session_state.client.query_points(
            st.session_state.collection_name,
            query_embedding
        )
        
        # Generate response with Qwen
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": result['payload']['image']},
                    {"type": "text", "text": f"Answer the user query: {query}, based on the image provided."}
                ]
            }
        ]
        
        inputs = qwen_processor(messages, return_tensors="pt")
        output = qwen_model.generate(**inputs, max_new_tokens=128)
        response = qwen_processor.decode(output[0], skip_special_tokens=True)
        
        st.write(response)

else:
    st.info("Please upload a document to start chatting!")
