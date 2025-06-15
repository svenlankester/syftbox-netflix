import os
import jinja2
import json
from pathlib import Path
from fastsyftbox import FastSyftBox
from fastapi import Request
from fastapi.responses import HTMLResponse
from syft_core import Client as SyftboxClient
from syft_core import SyftClientConfig

APP_NAME = os.getenv("APP_NAME", "syftbox-netflix")
AGGREGATOR_DATASITE = os.getenv("AGGREGATOR_DATASITE")

config = SyftClientConfig.load()
client = SyftboxClient(config)
path_top5_dp = Path(client.datasite_path.parent / AGGREGATOR_DATASITE / "app_data" / APP_NAME / "shared" / "top5_series.json")
path_local_recommends = Path(client.config.data_dir) / "private" / APP_NAME / "profile_0" / "recommendations.json"

with open(path_top5_dp, "r", encoding="utf-8") as f:
    top5_dp = json.load(f)

with open(path_local_recommends, "r", encoding="utf-8") as f:
    local_recommends = json.load(f)


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
    
    top_series = sorted(top5_dp, key=lambda x: x["count"], reverse=True)[:4]    # limited to top 4 and not 5 (for better visibility)
    top_recommends = sorted(local_recommends, key=lambda x: x["raw_score"], reverse=True)[:4]    # limited to top 4 and not 5 (for better visibility)

    series_for_template = [
        {"name": item["name"], "img": item["img"]}
        for item in top_series
    ]

    recommends_for_template = [
        {"name": item["name"], "img": item["img"]}
        for item in top_recommends
    ]

    template = jinja2.Template(template_content)

    rendered_content = template.render(series=series_for_template, recommends=recommends_for_template)

    return HTMLResponse(rendered_content)