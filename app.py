import streamlit as st
import torch
import os
import tempfile
import io

from PIL import Image
import fitz  # PyMuPDF

# Qdrant
from qdrant_client import QdrantClient, models

# ColPali
from colpali_engine.models import ColPali, ColPaliProcessor
from transformers import ColPaliForRetrieval, ColPaliProcessor as ColPaliHFProcessor

# Qwen2.5-VL
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor


# ------------------------
#     CONFIG & INIT
# ------------------------
st.set_page_config(
    page_title="Multimodal RAG powered by Qwen2.5-VL + ColPali + Qdrant",
    layout="wide"
)

# Qdrant constants
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "qwen-colpali-multimodalRAG"
EMBEDDING_SIZE = 128  # Adjust if your model produces different dims

# 1) Qdrant client
@st.cache_resource
def init_qdrant():
    client = QdrantClient(url=QDRANT_URL)
    # Create collection if it doesn't exist
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=EMBEDDING_SIZE,
                distance=models.Distance.COSINE
            )
        )
    return client

# 2) ColPali for retrieval
@st.cache_resource
def init_colpali():
    # Option A: Use the HF interface for retrieval:
    # model_name = "vidore/colpali-v1.2-hf"
    # retrieval_model = ColPaliForRetrieval.from_pretrained(
    #     model_name,
    #     torch_dtype=torch.bfloat16,
    #     device_map="auto"
    # ).eval()
    # retrieval_processor = ColPaliHFProcessor.from_pretrained(model_name)

    # Option B: Use the official colpali_engine library:
    model_name = "vidore/colpali-v1.2"
    embed_model = ColPali.from_pretrained(model_name)
    embed_processor = ColPaliProcessor.from_pretrained(model_name)
    return embed_model, embed_processor

# 3) Qwen2.5-VL
@st.cache_resource
def init_qwen():
    model_path = "Qwen/Qwen2.5-VL-3B-Instruct"
    qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto"
    )
    qwen_processor = AutoProcessor.from_pretrained(model_path)
    return qwen_model, qwen_processor

# ------------------------
#   PDF -> IMAGES UTILS
# ------------------------
def pdf_to_images(pdf_file):
    """Convert PDF pages to a list of PIL images."""
    images = []
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_file.read())
        tmp.flush()
        doc = fitz.open(tmp.name)
        for page_index in range(len(doc)):
            page = doc[page_index]
            pix = page.get_pixmap()
            img_data = pix.tobytes("png")
            pil_img = Image.open(io.BytesIO(img_data)).convert("RGB")
            images.append(pil_img)
        doc.close()
    os.unlink(tmp.name)
    return images

# ------------------------
#  STORE EMBEDDINGS
# ------------------------
def store_image_embeddings(images, embed_model, embed_processor, client):
    """Embed each page image with ColPali, then store in Qdrant."""
    for i, pil_img in enumerate(images):
        # Get embedding
        processed = embed_processor(images=pil_img, return_tensors="pt")
        # For the official colpali_engine approach:
        embedding = embed_model.get_image_features(**processed).detach().cpu().numpy().flatten()
        # Upsert into Qdrant
        point = models.PointStruct(
            id=i,
            vector=embedding,
            payload={"page_index": i}  # store any metadata you want
        )
        client.upsert(collection_name=COLLECTION_NAME, points=[point])

# ------------------------
#   RETRIEVE NEAREST
# ------------------------
def retrieve_best_match(query_text, embed_model, embed_processor, client):
    """Embed query, retrieve nearest image embedding from Qdrant, return the best payload."""
    # Get query embedding
    processed_q = embed_processor(text=query_text, return_tensors="pt")
    q_emb = embed_model.get_text_features(**processed_q).detach().cpu().numpy().flatten()
    # Search in Qdrant
    search_result = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=q_emb,
        limit=1  # top-1
    )
    if not search_result:
        return None
    # Return the top match
    return search_result[0]

# ------------------------
#   QWEN GENERATION
# ------------------------
def generate_answer(query, matched_payload, qwen_model, qwen_processor, images):
    """
    matched_payload should have 'id' or 'page_index' so we can fetch the image from images list.
    We'll feed the text query + that page image into Qwen2.5-VL.
    """
    if matched_payload is None:
        return "No matching page found in Qdrant."
    page_index = matched_payload.id
    if page_index < 0 or page_index >= len(images):
        return "Could not locate the matched page image."

    matched_image = images[page_index]

    # Qwen expects a list of messages with "role" and "content" array
    # Each content piece can be a dict with {"type":"text"/"image", ...}
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": query},
                {"type": "image", "image": matched_image}  # pass PIL image
            ]
        }
    ]
    # Prepare input
    inputs = qwen_processor(
        messages,
        return_tensors="pt"
    ).to(qwen_model.device)

    # Generate
    with torch.no_grad():
        output_ids = qwen_model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
    response = qwen_processor.decode(output_ids[0], skip_special_tokens=True)
    return response

# ------------------------
#    MAIN APP
# ------------------------
def main():
    st.title("Multimodal RAG powered by Qwen2.5-VL + ColPali + Qdrant")
    st.write("Upload a PDF, then ask questions about it.")

    # Initialize everything
    client = init_qdrant()
    embed_model, embed_processor = init_colpali()
    qwen_model, qwen_processor = init_qwen()

    # PDF upload
    pdf_file = st.file_uploader("Upload PDF", type=["pdf"])
    if pdf_file:
        if st.button("Index PDF"):
            with st.spinner("Converting PDF to images & embedding..."):
                # Convert PDF pages to images
                images = pdf_to_images(pdf_file)
                # Store embeddings
                store_image_embeddings(images, embed_model, embed_processor, client)
            st.success("PDF indexed successfully. You can now query it.")

            # Keep the images in session state for retrieval
            st.session_state["pdf_images"] = images

    # Query
    if "pdf_images" in st.session_state:
        query_text = st.text_input("Ask a question about your PDF:")
        if query_text:
            matched = retrieve_best_match(query_text, embed_model, embed_processor, client)
            answer = generate_answer(query_text, matched, qwen_model, qwen_processor, st.session_state["pdf_images"])
            st.markdown(f"**Answer:** {answer}")

    st.markdown("---")
    st.markdown("**Note:** This demo requires Qdrant running locally and the dev version of Transformers for Qwen2.5-VL.")


if __name__ == "__main__":
    main()
