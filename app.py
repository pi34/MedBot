import os
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tensorflow.keras.applications.resnet50 import preprocess_input

WOUND_MODEL_PATH = "Wound_model.h5"      
WOUND_DATA_DIR   = "ENTER" 
LLM_PATH         = "./medbot_finetuned"    
DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WOUND_KEYWORDS   = [
    "wound","bruise","burn","cut","abrasion","laceration",
    "pressure","diabetic","surgical","venous", "skin"
]
MAX_LLMTOKENS    = 512

@st.cache(allow_output_mutation=True)
def load_wound_model():
    
    model = tf.keras.models.load_model(WOUND_MODEL_PATH)
    
    classes = sorted(
        d for d in os.listdir(WOUND_DATA_DIR)
        if os.path.isdir(os.path.join(WOUND_DATA_DIR, d))
    )
    return model, classes

@st.cache(allow_output_mutation=True)
def load_llm():
    tokenizer = AutoTokenizer.from_pretrained(LLM_PATH, use_fast=True)
    model     = AutoModelForCausalLM.from_pretrained(LLM_PATH).to(DEVICE)
   
    if tokenizer.pad_token is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
        tokenizer.add_special_tokens({"pad_token":"[PAD]"})
        model.resize_token_embeddings(len(tokenizer))
    return tokenizer, model

wound_model, wound_classes = load_wound_model()
tokenizer, llm_model       = load_llm()

def classify_wound(img: Image.Image):
    img = img.resize((224,224))
    arr = np.array(img)
    x = preprocess_input(arr)
    x = np.expand_dims(x, 0)
    preds = wound_model.predict(x)
    idx = int(np.argmax(preds, axis=1)[0])
    return wound_classes[idx]

def generate_reply(history, classification=None):
    prompt = ""
    for speaker, text in history:
        tag = "Patient" if speaker=="Patient" else "Doctor"
        prompt += f"{tag}: {text}{tokenizer.eos_token}"
    if classification:
        prompt += f"Classification: The image appears to be a {classification}.{tokenizer.eos_token}"
    prompt += "Doctor:"
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LLMTOKENS
    ).to(DEVICE)
    out = llm_model.generate(
        **inputs,
        max_new_tokens=100,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
        no_repeat_ngram_size=2
    )
    gen = out[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()

# Streamlit UI 

st.title("MedBot: Medical Assistant with Wound Classification")

if "history" not in st.session_state:
    st.session_state.history = []
if "awaiting_image" not in st.session_state:
    st.session_state.awaiting_image = False

# Input form
with st.form("chat_form", clear_on_submit=True):
    user_msg  = st.text_input("You (Patient):")
    submitted = st.form_submit_button("Send")

if submitted and user_msg:
    st.session_state.history.append(("Patient", user_msg))

    # if patient mentions a wound-like keyword, ask for image
    if any(kw in user_msg.lower() for kw in WOUND_KEYWORDS):
        st.session_state.awaiting_image = True
        bot_msg = "I’m sorry to hear that. Could you please upload a photo of the area so I can help identify it?"
    else:
        bot_msg = generate_reply(st.session_state.history)
    st.session_state.history.append(("Doctor", bot_msg))


for speaker, text in st.session_state.history:
    if speaker=="Patient":
        st.markdown(f"**You:** {text}")
    else:
        st.markdown(f"**Doctor:** {text}")

# Image Stuff
if st.session_state.awaiting_image:
    uploaded = st.file_uploader("Upload wound image", type=["jpg","jpeg","png"])
    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Your upload", use_column_width=True)
        cls = classify_wound(image)
        st.success(f"Classification result: **{cls}**")
        
        # follow-up with classification context
        follow_up = generate_reply(st.session_state.history, classification=cls)
        st.session_state.history.append(("Doctor", follow_up))
        st.session_state.awaiting_image = False
