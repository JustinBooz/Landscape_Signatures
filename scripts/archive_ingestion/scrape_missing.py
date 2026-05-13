"""
Scrape missing Swiss H3 cells using the same multi-source pipeline.
Targets 433 cells that intersect Switzerland's boundary but were excluded
from the original scraping grid.
"""

import os
import asyncio
import aiohttp
import numpy as np
import h3
import json
import yaml
import webdataset as wds
import io
import math
import random
import sys
import time
import logging
from collections import defaultdict
import shapely.geometry
from concurrent.futures import ProcessPoolExecutor

# Reuse all the fetch/processing functions from unified_ingestion
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from baukultur_vpr.data.unified_ingestion import (
    worker_init, fetch_apple, fetch_google, fetch_bing, fetch_mapillary,
    haversine, is_in_switzerland, robust_get, download_image
)
from streetlevel import lookaround

# --- LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("scrape_missing.log", mode='a')
    ]
)
logger = logging.getLogger()


def load_config():
    for p in ["baukultur_vpr/config.yaml", "config.yaml", "../config.yaml"]:
        if os.path.exists(p):
            with open(p, "r") as f:
                return yaml.safe_load(f)
    return {}


async def process_cell(cid, session, auth, writer, api_sem, dl_sem, write_lock, 
                       token, state, executor):
    """Process a single H3 cell — identical logic to unified_ingestion."""
    if cid in state["processed_cells"]:
        return True, 0

    children = list(h3.cell_to_children(cid, 10))
    random.shuffle(children)

    total_found = 0
    seen = set()

    async def probe(c):
        nonlocal total_found
        async with api_sem:
            lat, lon = h3.cell_to_latlng(c)
            # Don't filter by boundary — we want border cells too
            srcs = ["apple", "google", "bing", "mapillary"]
            random.shuffle(srcs)
            found_something = False
            for s in srcs:
                pairs = []
                if s == "apple":
                    pairs = await fetch_apple(lat, lon, session, auth, dl_sem, executor)
                elif s == "google":
                    pairs = await fetch_google(lat, lon, session, executor)
                elif s == "bing":
                    pairs = await fetch_bing(lat, lon, session, executor)
                elif s == "mapillary":
                    pairs = await fetch_mapillary(lat, lon, session, token, dl_sem, executor)
                if pairs:
                    found_something = True
                    async with write_lock:
                        for p in pairs:
                            pair_key = (p["pano_id"], p["suffix"])
                            if pair_key in seen:
                                continue
                            writer.write({
                                "__key__": f"fill_{p['pano_id']}_{p['suffix']}",
                                "img1.jpg": p["bytes_a"],
                                "img2.jpg": p["bytes_b"],
                                "meta.json": json.dumps({
                                    "lat": p["lat"], "lon": p["lon"],
                                    "date": p["date"], "source": p["source"],
                                    "is_ugc": p["is_ugc"], "pano_id": p["pano_id"],
                                    "suffix": p["suffix"]
                                }).encode("utf-8")
                            })
                            logger.info(f"  -> [COMMIT] fill_{p['pano_id']}_{p['suffix']} ({p['source']})")
                            seen.add(pair_key)
                            total_found += 1
            return found_something

    # Reconnaissance phase
    seed = children[:20]
    rest = children[20:]
    seed_results = await asyncio.gather(*[probe(c) for c in seed])
    seed_hits = sum(1 for r in seed_results if r)

    # Deep patrol if sparse
    if seed_hits == 0 and len(rest) > 0:
        logger.info(f"[DEEP-PATROL] Cell {cid}: expanding by 80 probes")
        patrol = rest[:80]
        rest = rest[80:]
        patrol_results = await asyncio.gather(*[probe(c) for c in patrol])
        seed_hits += sum(1 for r in patrol_results if r)

    # Adaptive branching
    depth_ratio = max(0.25, seed_hits / 20.0)
    rng = random.Random(int(cid, 16))
    targets = [c for c in rest if rng.random() < depth_ratio]

    logger.info(f"[SMART] Cell {cid}: {seed_hits}/20 seed hits, depth={int(depth_ratio*100)}%, "
                f"probing {len(targets)} more")

    if targets:
        await asyncio.gather(*[probe(c) for c in targets])

    state["processed_cells"].add(cid)
    async with write_lock:
        with open("fill_ingestion_state.json", "w") as f:
            json.dump(list(state["processed_cells"]), f)

    logger.info(f"Finished Cell {cid}: {total_found} pairs")
    return True, total_found


async def main():
    cfg = load_config()
    token = cfg["api"]["mapillary_access_token"]

    # Load missing cells
    missing_coords = np.load("baukultur_vpr/data/h3_coords_missing_ch.npy")
    logger.info(f"{'='*60}")
    logger.info(f"SCRAPE MISSING SWISS CELLS")
    logger.info(f"{'='*60}")
    logger.info(f"Cells to scrape: {len(missing_coords)}")

    # Resume state
    state = {"processed_cells": set()}
    if os.path.exists("fill_ingestion_state.json"):
        with open("fill_ingestion_state.json") as f:
            state["processed_cells"] = set(json.load(f))
        logger.info(f"Resuming: {len(state['processed_cells'])} cells already done")

    # Output shards — continue from existing shard numbering
    shard_dir = "baukultur_vpr/data/shards"
    existing_shards = [f for f in os.listdir(shard_dir) if f.endswith(".tar")]
    start_shard = max([int(s.split("-")[-1].split(".")[0]) for s in existing_shards]) + 1 if existing_shards else 0
    logger.info(f"Starting at shard index {start_shard}")

    with ProcessPoolExecutor(max_workers=min(os.cpu_count() - 2, 20), 
                              initializer=worker_init) as executor:
        writer = wds.ShardWriter(
            f"{shard_dir}/dataset-unified-%06d.tar",
            maxsize=1e9, maxcount=5000, start_shard=start_shard
        )
        auth = lookaround.Authenticator()
        api_sem = asyncio.Semaphore(200)
        dl_sem = asyncio.Semaphore(400)
        write_lock = asyncio.Lock()
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64... Safari/537.36"}

        async with aiohttp.ClientSession(headers=headers, 
                                          connector=aiohttp.TCPConnector(limit=0)) as session:
            active_tasks = set()
            max_concurrent = 30
            total_pairs = 0

            for i, (lat, lon) in enumerate(missing_coords):
                cid = h3.latlng_to_cell(lat, lon, 7)
                if cid in state["processed_cells"]:
                    continue

                if len(active_tasks) >= max_concurrent:
                    done, active_tasks = await asyncio.wait(
                        active_tasks, return_when=asyncio.FIRST_COMPLETED
                    )
                    for d in done:
                        try:
                            _, pairs = d.result()
                            total_pairs += pairs
                        except:
                            pass

                task = asyncio.create_task(
                    process_cell(cid, session, auth, writer, api_sem, dl_sem,
                                 write_lock, token, state, executor)
                )
                active_tasks.add(task)

                if (i + 1) % 50 == 0:
                    logger.info(f"Progress: {i+1}/{len(missing_coords)} cells queued, "
                                f"{len(state['processed_cells'])} done, "
                                f"{total_pairs} pairs total")

            if active_tasks:
                results = await asyncio.gather(*active_tasks, return_exceptions=True)
                for r in results:
                    if isinstance(r, tuple):
                        total_pairs += r[1]

        writer.close()

    logger.info(f"{'='*60}")
    logger.info(f"SCRAPING COMPLETE: {len(state['processed_cells'])} cells, {total_pairs} pairs")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    asyncio.run(main())
