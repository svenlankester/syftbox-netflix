import os
from pathlib import Path
from fastsyftbox import FastSyftBox
from fastapi import Request
from fastapi.responses import HTMLResponse

APP_NAME = os.getenv("APP_NAME", "syftbox-netflix")

app = FastSyftBox(
    app_name=APP_NAME,
    syftbox_endpoint_tags=[
        "syftbox"
    ],  # endpoints with this tag are also available via Syft RPC
    include_syft_openapi=True,  # Create OpenAPI endpoints for syft-rpc routes
)

# Reference: https://github.com/madhavajay/youtube-wrapped/blob/main/app.py | https://github.com/openmined/fastsyftbox
# normal fastapi
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def ui_home(request: Request):
    current_dir = Path(__file__).parent
    template_path = current_dir / "participant_utils" / "home.html"
    with open(template_path, encoding="utf-8") as f:
        template_content = f.read()
        
    return HTMLResponse(template_content)