import streamlit as st
import pandas as pd

from analysis import (
    load_data,
    suggest_prompts,
    prompt_to_code,
    run_code,
    ask_llm
)

# ----------------- Page config -----------------

st.set_page_config(
    page_title="Personal AI Data Analyst",
    layout="wide"
)

st.title("🧠 Personal AI Data Analyst — Interactive Dashboard")

# ----------------- Sidebar -----------------

st.sidebar.header("Settings")

use_llm = st.sidebar.checkbox(
    "Use API (OpenRouter) for custom prompts",
    value=True
)

st.sidebar.markdown(
    """
    - Suggested prompts run **without API**
    - Custom prompts use **OpenRouter API**
    """
)

# ----------------- File upload -----------------

uploaded = st.file_uploader(
    "Upload CSV, Excel, or JSON",
    type=["csv", "xls", "xlsx", "json"]
)

if uploaded is None:
    st.info("⬆️ Upload a file to get started")
    st.stop()

# ----------------- Load data -----------------

try:
    df = load_data(uploaded)
except Exception as e:
    st.error(f"Failed to load file: {e}")
    st.stop()

st.success("✅ File loaded successfully")

with st.expander("Preview data (first 100 rows)"):
    st.dataframe(df.head(100))

# ----------------- Prompt selection -----------------

suggestions = suggest_prompts(df)

st.markdown("## 🧪 Choose an analysis or write your own")

col1, col2 = st.columns([3, 1])

with col1:
    selected_prompt = st.selectbox(
        "Suggested prompts",
        options=suggestions
    )

    custom_prompt = st.text_area(
        "Or write a custom prompt",
        height=100,
        placeholder="Example: Show top 5 rows sorted by sales"
    )

with col2:
    st.markdown("### ℹ️ Tips")
    st.write(
        "- Custom prompt → API\n"
        "- Suggested prompt → Fast & local"
    )

# ----------------- Final prompt -----------------

final_prompt = (
    custom_prompt.strip()
    if custom_prompt and custom_prompt.strip()
    else selected_prompt
)

st.markdown("### 📝 Final prompt")
st.code(final_prompt)

# ----------------- Run analysis -----------------

if st.button("▶️ Run analysis"):
    with st.spinner("Running analysis..."):

        # 1️⃣ Try deterministic prompt → code
        code = prompt_to_code(final_prompt, df)

        if code:
            res = run_code(df, code)

        else:
            # 2️⃣ Custom prompt → API
            if not use_llm:
                st.error(
                    "This is a custom prompt.\n"
                    "Enable 'Use API (OpenRouter)' from sidebar."
                )
                st.stop()

            # Send prompt to API
            llm_output = ask_llm(final_prompt)

            if llm_output.startswith("[API-error]") or llm_output.startswith("[API-failed]"):
                st.error("API call failed")
                st.code(llm_output)
                st.stop()

            # Extract python code block
            if "```python" not in llm_output:
                st.error("API did not return Python code")
                st.code(llm_output)
                st.stop()

            try:
                code = llm_output.split("```python")[1].split("```")[0]
                res = run_code(df, code)
            except Exception as e:
                st.error(f"Failed to execute generated code: {e}")
                st.code(llm_output)
                st.stop()

    # ----------------- Show result -----------------

    if res["type"] == "text":
        st.markdown("### 📄 Output")
        st.text(res["output"])

    elif res["type"] == "dataframe":
        st.markdown("### 📊 Table")
        st.dataframe(res["df"])

        csv = res["df"].to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download CSV",
            data=csv,
            file_name="result.csv",
            mime="text/csv"
        )

    elif res["type"] == "image":
        st.markdown("### 📈 Chart")
        st.image(res["path"], use_column_width=True)

    else:
        st.write("Unknown result:", res)
