import streamlit as st
import pickle
import numpy as np
import time

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="AI Quote Generator",
    page_icon="✨",
    layout="wide"
)

# =========================
# CUSTOM CSS
# =========================

st.markdown("""
<style>

.stApp{
    background: linear-gradient(
        135deg,
        #0f172a,
        #111827,
        #1e293b
    );
}

/* Hero Title */
.hero-title{
    text-align:center;
    font-size:65px;
    font-weight:bold;
    color:#00E5FF;
}

.hero-sub{
    text-align:center;
    font-size:20px;
    color:#cbd5e1;
    margin-bottom:30px;
}

/* Glass Effect */
.glass{
    background: rgba(255,255,255,0.05);
    border-radius:20px;
    padding:25px;
    backdrop-filter: blur(10px);
    border:1px solid rgba(255,255,255,0.1);
    margin-top:10px;
}

/* Quote Card */
.quote-box{
    background: rgba(0,229,255,0.08);
    border-left:5px solid #00E5FF;
    padding:25px;
    border-radius:15px;
    font-size:24px;
    color:white;
    line-height:1.8;
}

/* Footer */
.footer{
    text-align:center;
    color:#94a3b8;
    margin-top:40px;
}

</style>
""", unsafe_allow_html=True)

# =========================
# LOAD MODEL
# =========================

@st.cache_resource
def load_assets():

    model = load_model("lstm_model.h5")

    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    return model, tokenizer

model, tokenizer = load_assets()

# =========================
# VOCAB
# =========================

word_index = tokenizer.word_index

index_to_word = {
    index: word
    for word, index in word_index.items()
}

MAX_LEN = 10

# =========================
# GENERATOR
# =========================

def predict_next_word(text):

    sequence = tokenizer.texts_to_sequences([text])[0]

    sequence = pad_sequences(
        [sequence],
        maxlen=MAX_LEN,
        padding='pre'
    )

    prediction = model.predict(
        sequence,
        verbose=0
    )

    predicted_index = np.argmax(prediction)

    return index_to_word.get(
        predicted_index,
        ""
    )

def generate_quote(seed_text, n_words):

    output = seed_text

    for _ in range(n_words):

        next_word = predict_next_word(output)

        if next_word == "":
            break

        output += " " + next_word

    return output

# =========================
# SIDEBAR
# =========================

with st.sidebar:

    st.title("🤖 About")

    st.write("""
    ### AI Quote Generator

    Built using:

    ✅ TensorFlow

    ✅ LSTM Neural Network

    ✅ NLP

    ✅ Streamlit

    ---
    """)

    st.metric(
        "Vocabulary Size",
        len(word_index)
    )

    st.metric(
        "Model Type",
        "LSTM"
    )

    st.metric(
        "Framework",
        "TensorFlow"
    )

# =========================
# HERO SECTION
# =========================

st.markdown(
    """
    <div class="hero-title">
    ✨ AI Quote Generator
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="hero-sub">
    Generate inspirational quotes using a trained LSTM Neural Network
    </div>
    """,
    unsafe_allow_html=True
)

# =========================
# METRICS
# =========================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Model",
        "LSTM"
    )

with col2:
    st.metric(
        "Vocabulary",
        len(word_index)
    )

with col3:
    st.metric(
        "Status",
        "Online"
    )

# =========================
# EXAMPLES
# =========================

examples = [
    "Success comes",
    "Life is",
    "Dream big",
    "The future",
    "Believe in",
    "Hard work"
]

selected_example = st.selectbox(
    "✨ Try an example prompt",
    examples
)

# =========================
# INPUT
# =========================

seed = st.text_input(
    "✍ Enter Starting Words",
    value=selected_example
)

length = st.slider(
    "📏 Number of words to generate",
    min_value=5,
    max_value=50,
    value=20
)

# =========================
# HISTORY
# =========================

if "history" not in st.session_state:
    st.session_state.history = []

# =========================
# BUTTON
# =========================

if st.button(
    "🚀 Generate Quote",
    use_container_width=True
):

    if seed.strip() == "":

        st.warning(
            "Please enter some starting text."
        )

    else:

        with st.spinner(
            "🧠 AI is generating..."
        ):

            time.sleep(1.5)

            generated_text = generate_quote(
                seed,
                length
            )

        st.session_state.history.append(
            generated_text
        )

        st.success(
            "Quote Generated Successfully!"
        )

        st.markdown(
            f"""
            <div class="quote-box">
            {generated_text}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.download_button(
            "📥 Download Quote",
            generated_text,
            file_name="generated_quote.txt"
        )

# =========================
# HISTORY
# =========================

if len(st.session_state.history) > 0:

    st.markdown("## 📜 Previous Generations")

    for item in reversed(
        st.session_state.history[-5:]
    ):
        st.markdown(
            f"""
            <div class="glass">
            {item}
            </div>
            """,
            unsafe_allow_html=True
        )

# =========================
# FOOTER
# =========================

st.markdown(
    """
    <div class="footer">

    Made with ❤️ using TensorFlow, NLP and Streamlit

    </div>
    """,
    unsafe_allow_html=True
)