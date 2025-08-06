import streamlit as st
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

@st.cache_resource
def load_model():
    model_path = "./gpt2-recipes-manual_at_80pct"
    
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.to(device)
    model.eval()
    
    return model, tokenizer, device

def generate_recipe(model, tokenizer, device, prompt):
    formatted_prompt = f"{prompt} -> "
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
    
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
    return result

st.set_page_config(page_title="Recipe AI", page_icon="👨‍🍳")
st.title("👨‍🍳 Recipe AI Assistant")

model, tokenizer, device = load_model()
device_name = "Apple Silicon (MPS)" if device.type == "mps" else "CPU"

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