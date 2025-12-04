import streamlit as st
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from agent.llm import GeminiClient
from agent.functions import TutorFunctions

load_dotenv(override=True)  # prefer .env value in dev
st.set_page_config(page_title="Sinhala-English Tutor", page_icon="📚", layout="wide")

@st.cache_data
def load_vocab(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalize columns
    expected = ["sinhala", "english", "transliteration", "pos", "example_si", "example_en"]
    for col in expected:
        if col not in df.columns:
            df[col] = ""
    return df[expected]

DATA_PATH = Path("data/vocab.csv")
if not DATA_PATH.exists():
    st.warning("Sample vocabulary not found. Using embedded demo data.")
    demo = [
        {"sinhala": "කපු", "english": "brown", "transliteration": "kapu", "pos": "adj", "example_si": "කපු බළලා", "example_en": "brown cat"},
        {"sinhala": "පාසල්", "english": "school", "transliteration": "paasal", "pos": "noun", "example_si": "මම පාසල් යනවා", "example_en": "I go to school"},
        {"sinhala": "ආහාර", "english": "food", "transliteration": "aahara", "pos": "noun", "example_si": "ආහාර රසයි", "example_en": "Food is tasty"},
    ]
    vocab = pd.DataFrame(demo)
else:
    vocab = load_vocab(DATA_PATH)

functions = TutorFunctions(vocab)
st.title("Sinhala-English Tutor Agent")
st.caption("Learn English with Sinhala support: words, examples, and practice.")

# Sidebar controls
st.sidebar.header("Study Controls")
mode = st.sidebar.radio("Mode", ["Dictionary", "Flashcards", "Quiz"], index=0)
filter_pos = st.sidebar.multiselect("Part of speech", sorted(vocab["pos"].unique()))
query = st.sidebar.text_input("Search (Sinhala/English/Transliteration)")

# Filtering
filtered = functions.search(query)
if filter_pos:
    filtered = filtered[filtered["pos"].isin(filter_pos)]
if query:
    q = query.strip().lower()
    filtered = filtered[
        filtered["sinhala"].str.contains(q, case=False, na=False) |
        filtered["english"].str.contains(q, case=False, na=False) |
        filtered["transliteration"].str.contains(q, case=False, na=False)
    ]

if mode == "Dictionary":
    st.subheader("Dictionary")
    st.dataframe(
        filtered.rename(
            columns={
                "sinhala": "Sinhala",
                "english": "English",
                "transliteration": "Transliteration",
                "pos": "Part of Speech",
                "example_si": "Example (SI)",
                "example_en": "Example (EN)",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

elif mode == "Flashcards":
    st.subheader("Flashcards")
    if filtered.empty:
        st.info("No cards match your filters.")
    else:
        idx = st.session_state.get("card_idx", 0)
        if idx >= len(filtered):
            idx = 0
        row = filtered.iloc[idx]
        st.markdown(f"### {row['sinhala']} — {row['english']}")
        with st.expander("Show details"):
            st.write("Transliteration:", row["transliteration"]) 
            st.write("Part of speech:", row["pos"]) 
            st.write("Example (SI):", row["example_si"]) 
            st.write("Example (EN):", row["example_en"]) 
        # Gemini explanation
        try:
            gem = GeminiClient()
            exp = gem.explain_word(row["sinhala"], row["english"]).get("explanation", "")
            st.info(exp)
        except Exception as e:
            st.caption(f"LLM unavailable: {e}")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Previous"):
                st.session_state["card_idx"] = max(0, idx - 1)
                st.experimental_rerun()
        with col2:
            st.write("")
        with col3:
            if st.button("Next"):
                st.session_state["card_idx"] = min(len(filtered) - 1, idx + 1)
                st.experimental_rerun()

elif mode == "Quiz":
    st.subheader("Quiz: Sinhala → English")
    if filtered.empty:
        st.info("No words to quiz on. Adjust filters.")
    else:
        # Pick a deterministic item from filtered for simplicity
        quiz_idx = st.session_state.get("quiz_idx", 0) % len(filtered)
        row = filtered.iloc[quiz_idx]
        st.write("Word:", row["sinhala"])
        answer = st.text_input("Type the English meaning")
        if st.button("Check"):
            correct = row["english"].strip().lower()
            user = (answer or "").strip().lower()
            if user == correct:
                st.success("Correct! 🎉")
            else:
                st.error(f"Not quite. Answer: {row['english']}")
        if st.button("Next Question"):
            st.session_state["quiz_idx"] = quiz_idx + 1
            st.experimental_rerun()
        st.divider()
        # Generate extra quiz via Gemini
        try:
            gem = GeminiClient()
            items = functions.sample_items(8)
            quiz = gem.generate_quiz(items, n=5)
            st.json(quiz)
        except Exception as e:
            st.caption(f"LLM unavailable: {e}")

st.divider()
st.markdown("""
### Upload your vocabulary
Upload a CSV file with columns: `sinhala,english,transliteration,pos,example_si,example_en`.
"""
)
uploaded = st.file_uploader("Upload CSV", type=["csv"])
if uploaded is not None:
    try:
        user_df = pd.read_csv(uploaded)
        st.success(f"Loaded {len(user_df)} entries from uploaded file.")
        vocab = TutorFunctions.normalize(user_df)
        functions = TutorFunctions(vocab)
        st.dataframe(vocab, use_container_width=True)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
