import re
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"
INDEX_FILE = FRONTEND_DIST / "index.html"


def load_frontend_html() -> str:
    if not INDEX_FILE.is_file():
        raise FileNotFoundError(
            "The React build is missing. Build it with `cd frontend && npm ci && npm run build`, "
            "then commit the frontend/dist folder."
        )

    document = INDEX_FILE.read_text(encoding="utf-8")

    def resolve_asset(url: str) -> Path:
        if not url.startswith("/assets/"):
            raise FileNotFoundError(f"Unsupported frontend asset path: {url}")
        asset_path = (FRONTEND_DIST / url.lstrip("/")).resolve()
        if FRONTEND_DIST.resolve() not in asset_path.parents or not asset_path.is_file():
            raise FileNotFoundError(f"Frontend build asset not found: {url}")
        return asset_path

    def inline_script(match: re.Match[str]) -> str:
        asset_url = match.group("url")
        if not asset_url.endswith(".js"):
            return match.group(0)
        script = resolve_asset(asset_url).read_text(encoding="utf-8")
        script = re.sub(r"</script", r"<\/script", script, flags=re.IGNORECASE)
        return f'<script type="module">{script}</script>'

    def inline_stylesheet(match: re.Match[str]) -> str:
        asset_url = match.group("url")
        if not asset_url.endswith(".css"):
            return match.group(0)
        stylesheet = resolve_asset(asset_url).read_text(encoding="utf-8")
        return f"<style>{stylesheet}</style>"

    document = re.sub(
        r'<script\b[^>]*\bsrc="(?P<url>[^"]+)"[^>]*>\s*</script>',
        inline_script,
        document,
        flags=re.IGNORECASE,
    )
    return re.sub(
        r'<link\b(?=[^>]*\brel="stylesheet")[^>]*\bhref="(?P<url>[^"]+)"[^>]*>',
        inline_stylesheet,
        document,
        flags=re.IGNORECASE,
    )


st.set_page_config(page_title="AI Interior Design", layout="wide")

try:
    frontend_html = load_frontend_html()
except FileNotFoundError as error:
    st.error(str(error), title="Frontend build unavailable")
    st.stop()

st.iframe(
    frontend_html,
    height=900,
    alt="AI virtual interior design frontend",
)
