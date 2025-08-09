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
from datetime import date

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

def update_inventory_from_prompt(prompt):
    missing_items = []

    for item in ITEMS_LIST:
        if item.lower() in prompt.lower():
            # Deduct 1 by default for matched ingredient
            found = False
            for entry in st.session_state.fridge:
                if entry["item"] == item:
                    entry["quantity"] -= 1
                    if entry["quantity"] < 0:
                        missing_items.append(item)
                    found = True
                    break
            if not found:
                missing_items.append(item)

    # Remove items with zero or less quantity
    st.session_state.fridge = [f for f in st.session_state.fridge if f["quantity"] > 0]

    # Save missing items to session state for display after rerun
    st.session_state.missing_items = missing_items


st.set_page_config(page_title="Recipe AI", page_icon="👨‍🍳")
st.title("👨‍🍳 Recipe AI Assistant")

# --------------------------
# 1. Initialize Fridge State
# --------------------------
if "fridge" not in st.session_state:
    st.session_state.fridge = []  # list of dicts: {item, quantity, expiry}

# --------------------------
# 2. Predefined Items
# --------------------------
ITEMS_LIST = [
    "milk", "cheese", "cream cheese", "butter", "yogurt", "sour cream", "egg",
    "tomato", "onion", "lettuce", "spinach", "cabbage", "zucchini", "carrot",
    "garlic", "cucumber", "pepper", "celery", "apple", "grapes", "corn", "mushroom"
]

# --------------------------
# 3. Fridge Inventory UI
# --------------------------
# Initialize state flags
if "recipe_generated" not in st.session_state:
    st.session_state.recipe_generated = False
if "accept_reject_used" not in st.session_state:
    st.session_state.accept_reject_used = False

#defult item and quantity for quick add
if "fridge_initialized" not in st.session_state:
    st.session_state.fridge.append({
        "item": "milk",
        "quantity": 1,
        "expiry": date.today()
    })
    st.session_state.fridge.append({
        "item": "egg",
        "quantity": 6,
        "expiry": date.today()
    })
    st.session_state.fridge_initialized = True

with st.container():
    st.header("🥶 Fridge Inventory Tracker")

    col1, col2, col3, col4 = st.columns([3, 1, 2, 1])
    with col1:
        item = st.selectbox("Select Item", ITEMS_LIST)
    with col2:
        quantity = st.number_input("Quantity", min_value=1, step=1)
    with col3:
        expiry = st.date_input("Expiry Date", min_value=date.today())
    with col4:
        if st.button("➕ Add Item"):
            st.session_state.fridge.append({
                "item": item,
                "quantity": quantity,
                "expiry": expiry
            })
    # After the columns block, outside the column but still inside the container
    if st.session_state.fridge and len(st.session_state.fridge) > 0:
        # Show success only if the last item was just added (by checking if fridge was updated this run)
        last_entry = st.session_state.fridge[-1]
        # Optionally, you could track the last added item in session_state for more robust logic
        st.success(f"Added {last_entry['quantity']} x {last_entry['item']} (expires {last_entry['expiry']}) to fridge.")

    # --------------------------
    # 4. Display Current Inventory
    # --------------------------
    if st.session_state.fridge:
        st.subheader("Current Fridge Contents")
        st.table(st.session_state.fridge)
    else:
        st.info("Fridge is currently empty.")

model, tokenizer, device = load_model()
if device.type == "cuda":
    device_name = f"NVIDIA CUDA (GPU: {torch.cuda.get_device_name(0)})"
elif device.type == "mps":
    device_name = "Apple Silicon (MPS)"
else:
    device_name = "CPU"

if not st.session_state.recipe_generated:
    st.info(f"Model loaded on {device_name}")
st.write("Enter ingredients below and click 'Generate Recipe'")

with st.form("recipe_form", clear_on_submit=True):
    user_input = st.text_input("Your ingredients:", placeholder="e.g., egg, flour, sugar")
    submitted = st.form_submit_button("Generate Recipe")
    
    if submitted and user_input:
        with st.spinner("Generating recipe..."):
            try:
                recipe = generate_recipe(model, tokenizer, device, user_input)
                st.session_state.generated_prompt = user_input
                st.session_state.generated_recipe = recipe
                st.session_state.recipe_generated = True
                st.session_state.accept_reject_used = False  # reset for new recipe
                st.session_state.missing_items = []  # reset missing items

            except Exception as e:
                st.error(f"Error: {str(e)}")

if st.session_state.get("generated_recipe"):
    #st.write(st.session_state.generated_recipe)
    recipe = st.session_state.generated_recipe
    if "Instructions:" in recipe:
        parts = recipe.split("Instructions:", 1)
        if len(parts) == 2:
            title = parts[0].strip().rstrip('.')
            instructions = parts[1].strip()
            if not st.session_state.get("accept_reject_used"):
                st.success("Recipe Generated!")
            st.markdown(f"**{title}**")
            st.markdown(f"**Instructions:** {instructions}")
        else:
            if not st.session_state.get("accept_reject_used"):
                st.success("Recipe Generated!")
            st.markdown(recipe)
    else:
        if not st.session_state.get("accept_reject_used"):
            st.success("Recipe Generated!")
        st.markdown(f"**Recipe Suggestion:** {recipe}")

if st.session_state.get("missing_items"):
    st.warning("🛒 You need to buy: " + ", ".join(st.session_state.missing_items))

if st.session_state.get("recipe_generated"):
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Accept Recipe", disabled=st.session_state.accept_reject_used):
            update_inventory_from_prompt(st.session_state.generated_prompt)
            st.success("Recipe accepted. Matching ingredients deducted from fridge.")
            st.session_state.accept_reject_used = True
            st.rerun()

    with col2:
        if st.button("❌ Reject Recipe", disabled=st.session_state.accept_reject_used):
            st.info("Recipe rejected. No changes made to fridge.")
            st.session_state.accept_reject_used = True
            st.rerun()

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