"""
7B Embedding Map Backend
========================
FastAPI server providing:
  - Image streaming from tar shards
  - Metadata lookup via KDTree
  - CORS-enabled for frontend dev
"""

import os
import tarfile
import json
import pandas as pd
from io import BytesIO
from scipy.spatial import cKDTree
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="7B Landscape Signatures API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SHARDS_DIR = os.path.abspath(os.path.join(BASE_DIR, "../baukultur_vpr/data/shards"))
FRONTEND_DIR = os.path.abspath(os.path.join(BASE_DIR, "../map_7b_frontend"))

# ---- Data loading ----
df = None
kdtree = None

def load_data():
    global df, kdtree
    if df is not None:
        return
    print("Loading data for KDTree...")
    # Data is now in data/ subfolder for static serving, but we load it from disk here
    parquet_path = os.path.join(FRONTEND_DIR, "data_7b.parquet")
    df = pd.read_parquet(parquet_path)
    tree_points = df[['lon', 'lat']].values
    kdtree = cKDTree(tree_points)

@app.on_event("startup")
async def startup_event():
    load_data()

# ---- Search endpoints ----

@app.get("/closest_image")
async def get_closest_image(lat: float, lon: float):
    # Find nearest point in KDTree
    dist, idx = kdtree.query([[lon, lat]], k=1)
    idx = idx[0]
    
    # Return metadata
    row = df.iloc[idx]
    return {
        "index": int(idx),
        "distance": float(dist[0]),
        "lat": float(row['lat']),
        "lon": float(row['lon']),
        "cluster_micro": int(row['cluster_micro']),
        "cluster_meso": int(row['cluster_meso']),
        "cluster_macro": int(row['cluster_macro']),
        "tar_file": str(row['tar_file']),
        "image_id": str(row['image_id'])
    }

# ---- Image endpoint ----

TAR_CACHE = {}

@app.get("/image/{tar_name}/{id1}")
async def get_image(tar_name: str, id1: str):
    tar_path = os.path.join(SHARDS_DIR, tar_name)
    if not os.path.exists(tar_path):
        raise HTTPException(status_code=404, detail="Tar not found")

    try:
        if tar_path not in TAR_CACHE:
            print(f"Opening and indexing tar: {tar_name}")
            t = tarfile.open(tar_path, "r:")
            TAR_CACHE[tar_path] = t
        else:
            t = TAR_CACHE[tar_path]

        img_member = None
        root_id = id1.split('_')[0]
        
        for member in t.getmembers():
            if (id1 in member.name or root_id in member.name) and member.name.endswith(".img1.jpg"):
                img_member = member
                break

        if img_member is None:
            raise HTTPException(status_code=404, detail="Image member not found in tar")

        f = t.extractfile(img_member)
        if not f:
            raise HTTPException(status_code=500, detail="Could not extract file")

        buffer = BytesIO(f.read())
        return StreamingResponse(buffer, media_type="image/jpeg")

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


# ---- Serve frontend static files ----
# StaticFiles will automatically serve everything in FRONTEND_DIR,
# including the new data/ subfolder.
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=False, workers=1)
