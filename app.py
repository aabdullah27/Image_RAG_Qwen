import streamlit as st
import torch
import os
import tempfile
import io
from PIL import Image
import fitz  # PyMuPDF

# Qdrant
from qdrant_client import QdrantClient, models
from qdrant_client.conversions.common_types import Record

# ColPali from Hugging Face Transformers
from transformers import ColPaliForRetrieval, ColPaliProcessor as HFColPaliProcessor

# Qwen2.5-VL
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor


# ------------------------
#   CONFIG & CONSTANTS
# ------------------------
st.set_page_config(
    page_title="Multimodal RAG: Qwen2.5-VL + ColPali + Qdrant",
    layout="wide"
)

QDRANT_URL = "http://localhost:6333"  # Change if needed
COLLECTION_NAME = "qwen-colpali-multimodalRAG"
EMBEDDING_SIZE = 128  # Adjust if your model has different dimension

# ------------------------
#    QDRANT INIT
# ------------------------
@st.cache_resource
def init_qdrant():
    """
    Attempt to connect to Qdrant. If unreachable, raise an exception with a
    helpful message.
    """
    try:
        client = QdrantClient(url=QDRANT_URL)
        # Simple test call: list collections
        all_collections = client.get_collections()
        st.info(f"Successfully connected to Qdrant at {QDRANT_URL}. Found collections: "
                f"{[c.name for c in all_collections.collections]}")

        # Create collection if not exists
        if not client.collection_exists(COLLECTION_NAME):
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=EMBEDDING_SIZE,
                    distance=models.Distance.COSINE
                )
            )
        return client

    except Exception as e:
        st.error(f"Could not connect to Qdrant at {QDRANT_URL}. "
                 "Check that Qdrant is running and accessible.")
        raise e

# ------------------------
#   COLPALI INIT
# ------------------------
@st.cache_resource
def init_colpali_hf():
    model_name = "vidore/colpali-v1.2-hf"  # or your chosen HF model
    retrieval_model = ColPaliForRetrieval.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    ).eval()
    retrieval_processor = HFColPaliProcessor.from_pretrained(model_name)
    return retrieval_model, retrieval_processor

# ------------------------
#   QWEN2.5-VL INIT
# ------------------------
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
#   PDF -> IMAGES
# ------------------------
def pdf_to_images(pdf_file):
    """Convert PDF pages to a list of PIL Images."""
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
def store_image_embeddings(images, retrieval_model, retrieval_processor, client):
    """
    Embed each page image with HF ColPali, then store in Qdrant.
    """
    for i, pil_img in enumerate(images):
        inputs = retrieval_processor(images=pil_img, return_tensors="pt")
        with torch.no_grad():
            emb = retrieval_model(**inputs).embeddings
        embedding = emb[0].detach().cpu().numpy().flatten()

        point = models.PointStruct(
            id=i,
            vector=embedding,
            payload={"page_index": i}
        )
        client.upsert(collection_name=COLLECTION_NAME, points=[point])

# ------------------------
#   RETRIEVE NEAREST
# ------------------------
def retrieve_best_match(query_text, retrieval_model, retrieval_processor, client):
    """
    Embed query, retrieve nearest image from Qdrant.
    """
    inputs = retrieval_processor(text=[query_text], return_tensors="pt")
    with torch.no_grad():
        emb = retrieval_model(**inputs).embeddings
    q_emb = emb[0].detach().cpu().numpy().flatten()

    result = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=q_emb,
        limit=1
    )
    if not result:
        return None
    return result[0]  # top match

# ------------------------
#   QWEN GENERATION
# ------------------------
def generate_answer(query, matched_payload, qwen_model, qwen_processor, images):
    """
    Use Qwen2.5-VL to answer question given the matched page image.
    """
    if matched_payload is None:
        return "No match found in Qdrant."

    page_index = matched_payload.id
    if page_index < 0 or page_index >= len(images):
        return "Could not locate the matched page image."

    matched_image = images[page_index]

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": query},
                {"type": "image", "image": matched_image}
            ]
        }
    ]
    inputs = qwen_processor(messages, return_tensors="pt").to(qwen_model.device)

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
#        MAIN APP
# ------------------------
def main():
    st.title("Multimodal RAG: Qwen2.5-VL + ColPali + Qdrant")
    st.write("Upload a PDF, embed with ColPali, store in Qdrant, and query with Qwen2.5-VL.")

    # Initialize Qdrant
    try:
        client = init_qdrant()
    except Exception:
        st.stop()  # Stop the app if Qdrant connection fails

    retrieval_model, retrieval_processor = init_colpali_hf()
    qwen_model, qwen_processor = init_qwen()

    pdf_file = st.file_uploader("Upload PDF", type=["pdf"])
    if pdf_file:
        if st.button("Index PDF"):
            with st.spinner("Indexing PDF..."):
                images = pdf_to_images(pdf_file)
                store_image_embeddings(images, retrieval_model, retrieval_processor, client)
            st.session_state["pdf_images"] = images
            st.success("PDF indexed! Ask questions below.")

    if "pdf_images" in st.session_state:
        query_text = st.text_input("Ask a question about your PDF:")
        if query_text:
            match = retrieve_best_match(query_text, retrieval_model, retrieval_processor, client)
            answer = generate_answer(query_text, match, qwen_model, qwen_processor, st.session_state["pdf_images"])
            st.write("**Answer:**", answer)

    st.markdown("---")
    st.markdown("Powered by Qwen2.5-VL, ColPali, Qdrant, and Streamlit")

if __name__ == "__main__":
    main()
