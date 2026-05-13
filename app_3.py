import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

st.set_page_config(
    page_title="Arabic Text Summarization",
    page_icon="📝",
    layout="centered"
)

st.markdown("""
    <style>
        .stTextArea textarea { font-size: 16px; }
        .stButton button { width: 100%; }
        .result-box {
            background-color: #1e1e1e;
            border-left: 4px solid #1f77b4;
            padding: 15px;
            border-radius: 8px;
            direction: rtl;
            font-size: 16px;
            line-height: 1.8;
        }
    </style>
""", unsafe_allow_html=True)

MODEL_PATH = r"C:\Users\LENOVO\Downloads\New folder"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
    model.to(DEVICE)
    model.eval()
    return tokenizer, model

def summarize(text, tokenizer, model, max_new_tokens=128, num_beams=4):
    inputs = tokenizer(
        "summarize: " + text,
        return_tensors="pt",
        max_length=512,
        truncation=True,
    ).to(DEVICE)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


st.title("📝 Arabic Text Summarization")
st.markdown("Enter an Arabic text and the model will generate an abstractive summary using **AraT5**.")
st.divider()

with st.spinner("Loading model..."):
    tokenizer, model = load_model()
st.success(f"✅ Model loaded — running on **{DEVICE.upper()}**")

text_input = st.text_area(
    "Input Text",
    height=250,
    placeholder="Enter Arabic text here...",
)

num_beams = 4
max_tokens = st.selectbox(
    "Max Summary Length",
    options=[50, 75, 100, 128, 150, 175, 200],
    index=3,
)

if st.button("🔍 Summarize", type="primary"):
    if not text_input.strip():
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Generating summary..."):
            summary = summarize(text_input, tokenizer, model, max_new_tokens=max_tokens, num_beams=num_beams)
        st.markdown("### Summary")
        st.markdown(f'<div class="result-box">{summary}</div>', unsafe_allow_html=True)
        st.divider()
        col1, col2 = st.columns(2)
        col1.metric("Input Word Count", len(text_input.split()))
        col2.metric("Summary Word Count", len(summary.split()))