# home.py
# A Streamlit app that uses a fine-tuned GPT-2 model to generate recipes based on user-provided ingredients.
# The app supports both CPU and Apple Silicon (MPS) devices.
# It includes example buttons for quick testing.
# Requires: streamlit, torch, transformers
# To run: streamlit run home.py
# Ensure you have the model directory "gpt2-recipes-manual_at_80pct" in the same directory as this script.
# activate ~/projects/ie7374/ie7374/bin/activate
# pip install streamlit torch transformers
# streamlit run home.py
import streamlit as st
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import re

@st.cache_resource
def load_model():
    model_path = "./gpt2-recipes-manual_at_80pct"
    
    # Select the best available device: CUDA > MPS > CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.to(device)
    model.eval()
    
    return model, tokenizer, device

def is_suspicious(text):
    return re.search(r'https?://', text, re.IGNORECASE) is not None

def generate_recipe(model, tokenizer, device, prompt, max_retries=3):
    formatted_prompt = f"{prompt} -> "
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)

    last_result = None
    
    for attempt in range(1, max_retries + 1):
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_length=120,
                do_sample=True,
                top_k=50,
                top_p=0.92,
                temperature=0.65,
                repetition_penalty=1.3,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        result = generated_text[len(formatted_prompt):].strip()
        last_result = result
        if not is_suspicious(result):
            return result
        else:
            st.warning(f"Suspicious content detected. Retrying... (Attempt {attempt}/{max_retries})")
            # Retry with the same prompt the model will generate a different output
    
    return result

st.set_page_config(page_title="Recipe AI", page_icon="👨‍🍳")
st.title("👨‍🍳 Recipe AI Assistant")

model, tokenizer, device = load_model()
if device.type == "cuda":
    device_name = f"NVIDIA CUDA (GPU: {torch.cuda.get_device_name(0)})"
elif device.type == "mps":
    device_name = "Apple Silicon (MPS)"
else:
    device_name = "CPU"

st.info(f"Model loaded on {device_name}")
st.write("Enter ingredients below and click 'Generate Recipe'")

with st.form("recipe_form", clear_on_submit=True):
    user_input = st.text_input("Your ingredients:", placeholder="e.g., egg, flour, sugar")
    submitted = st.form_submit_button("Generate Recipe")
    
    if submitted and user_input:
        with st.spinner("Generating recipe..."):
            try:
                recipe = generate_recipe(model, tokenizer, device, user_input)
                
                if "Instructions:" in recipe:
                    parts = recipe.split("Instructions:", 1)
                    if len(parts) == 2:
                        title = parts[0].strip().rstrip('.')
                        instructions = parts[1].strip()
                        st.success("Recipe Generated!")
                        st.markdown(f"**{title}**")
                        st.markdown(f"**Instructions:** {instructions}")
                    else:
                        st.success("Recipe Generated!")
                        st.markdown(recipe)
                else:
                    st.success("Recipe Generated!")
                    st.markdown(f"**Recipe Suggestion:** {recipe}")
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")

st.subheader("💡 Try these examples:")
col1, col2 = st.columns(2)

with col1:
    if st.button("🥞 egg, flour, sugar"):
        st.session_state.example = "egg, flour, sugar"
    if st.button("🧀 cheese, tomato, lettuce"):
        st.session_state.example = "cheese, tomato, lettuce"

with col2:
    if st.button("🍄 butter, garlic, mushroom"):
        st.session_state.example = "butter, garlic, mushroom"
    if st.button("🍎 apple, grapes, yogurt"):
        st.session_state.example = "apple, grapes, yogurt"

if "example" in st.session_state:
    example = st.session_state.example
    del st.session_state.example
    
    st.write(f"**Generating recipe for:** {example}")
    with st.spinner("Generating recipe..."):
        try:
            recipe = generate_recipe(model, tokenizer, device, example)
            
            if "Instructions:" in recipe:
                parts = recipe.split("Instructions:", 1)
                if len(parts) == 2:
                    title = parts[0].strip().rstrip('.')
                    instructions = parts[1].strip()
                    st.success("Recipe Generated!")
                    st.markdown(f"**{title}**")
                    st.markdown(f"**Instructions:** {instructions}")
                else:
                    st.success("Recipe Generated!")
                    st.markdown(recipe)
            else:
                st.success("Recipe Generated!")
                st.markdown(f"**Recipe Suggestion:** {recipe}")
                
        except Exception as e:
            st.error(f"Error: {str(e)}")