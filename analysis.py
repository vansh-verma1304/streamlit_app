# ================== IMPORTS ==================

import io
import os
import sys
import json
import textwrap
import tempfile
from pathlib import Path

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from dotenv import load_dotenv

# ================== LOAD ENV ==================

load_dotenv()  # .env file load karega

# ================== DATA LOADING ==================

def _looks_like_csv(raw_bytes: bytes) -> bool:
    try:
        sample = raw_bytes[:1024].decode(errors="ignore")
    except Exception:
        return False
    return "," in sample and "\n" in sample


def load_data(file_or_path) -> pd.DataFrame:
    if isinstance(file_or_path, (str, Path)):
        p = Path(file_or_path)
        ext = p.suffix.lower()

        if ext == ".csv":
            return pd.read_csv(p)
        if ext in [".xls", ".xlsx"]:
            return pd.read_excel(p)
        if ext == ".json":
            return pd.read_json(p)

    name = getattr(file_or_path, "name", "")
    suffix = Path(name).suffix.lower()
    raw = file_or_path.read()

    if isinstance(raw, str):
        raw = raw.encode("utf-8")

    bio = io.BytesIO(raw)

    if suffix == ".csv" or _looks_like_csv(raw):
        return pd.read_csv(bio)
    if suffix in [".xls", ".xlsx"]:
        return pd.read_excel(bio)
    if suffix == ".json":
        return pd.read_json(bio)

    raise ValueError("Unsupported file format")

# ================== PROMPT HELPERS ==================

def suggest_prompts(df: pd.DataFrame):
    prompts = [
        "dataset ke first 5 rows dikhao",
        "dataset ka short summary do",
        "numeric columns ka average batao",
        "missing values ka count dikhao",
    ]
    return prompts


def prompt_to_code(prompt: str, df: pd.DataFrame):
    p = prompt.lower().strip()

    if "first 5" in p:
        return "result = df.head(5)"

    if "summary" in p:
        return "result = df.describe(include='all')"

    if "missing" in p:
        return "result = df.isnull().sum()"

    return None  # custom prompt → API

# ================== CODE EXECUTION ==================

def run_code(df: pd.DataFrame, code: str):
    local_ns = {
        "df": df,
        "pd": pd,
        "np": np,
        "plt": plt
    }

    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()

    try:
        exec(code, {}, local_ns)

        if plt.get_fignums():
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f:
                plt.savefig(f.name, dpi=150, bbox_inches="tight")
                plt.close("all")
                return {"type": "image", "path": f.name}

        if "result" in local_ns:
            res = local_ns["result"]
            if isinstance(res, pd.DataFrame):
                return {"type": "dataframe", "df": res}
            else:
                return {"type": "text", "output": str(res)}

        output = buffer.getvalue().strip()
        return {"type": "text", "output": output or "No output"}

    except Exception as e:
        return {"type": "text", "output": f"Execution error: {e}"}

    finally:
        sys.stdout = old_stdout

# ================== OPENROUTER API ==================

def ask_llm(prompt: str, timeout: int = 60) -> str:
    """
    Custom prompt → OpenRouter API
    Response MUST contain ```python``` code
    """

    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        return "[API-error] OPENROUTER_API_KEY not found in environment"

    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        data=json.dumps({
            "model": "stepfun/step-3.5-flash:free",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a data analyst.\n"
                        "Return ONLY python code inside ```python```.\n"
                        "DataFrame name is df.\n"
                        "Use pandas and matplotlib only.\n"
                        "No explanations."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }),
        timeout=timeout
    )

    if response.status_code != 200:
        return f"[API-error] {response.text}"

    data = response.json()
    return data["choices"][0]["message"].get("content", "")
