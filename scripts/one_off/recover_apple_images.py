import os
import sys
import json
import logging
import asyncio
import aiohttp
from concurrent.futures import ProcessPoolExecutor
import webdataset as wds
from streetlevel import lookaround
import h3

# Insert local modules into path for importing target functions
sys.path.insert(0, os.path.abspath('baukultur_vpr/data'))
from unified_ingestion import fetch_apple

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
                    handlers=[logging.StreamHandler(), logging.FileHandler('apple_recovery.log')])
logger = logging.getLogger('Recovery')

async def process_recovery_cell(cid, session, auth, writer, dl_sem, executor, write_lock):
    """Process a single res-7 cell using the same strategy as the main pipeline:
    sample res-10 children, find coverage, then deep-extract at res-10."""
    children = list(h3.cell_to_children(cid, res=10))

    # Sample 20 points to probe for Apple coverage
    sample_stride = max(1, len(children) // 20)
    samples = children[::sample_stride]

    seen = set()
    total_recovered = 0

    async def recover_child(child_cid):
        nonlocal total_recovered
        try:
            lat, lon = h3.cell_to_latlng(child_cid)
            pairs = await fetch_apple(lat, lon, session, auth, dl_sem, executor)
            if not pairs:
                return

            for p in pairs:
                pair_key = (p["pano_id"], p["suffix"])
                if pair_key in seen:
                    continue
                seen.add(pair_key)

                async with write_lock:
                    writer.write({
                        "__key__": f"unified_{p['pano_id']}_{p['suffix']}",
                        "img1.jpg": p["bytes_a"],
                        "img2.jpg": p["bytes_b"],
                        "meta.json": json.dumps({"lat":p["lat"], "lon":p["lon"], "date":p["date"], "source":p["source"], "is_ugc":p["is_ugc"], "pano_id":p["pano_id"], "suffix":p["suffix"]}).encode("utf-8")
                    })
                total_recovered += 1
        except Exception as e:
            logger.debug(f"Recovery error on {child_cid}: {e}")

    # Probe samples first
    await asyncio.gather(*[recover_child(c) for c in samples])

    # If coverage found, deep-extract remaining children in batches
    if total_recovered > 0:
        logger.info(f"Coverage found in {cid}, deep extracting...")
        remaining = [c for c in children if c not in set(samples)]
        batch_size = 50
        for i in range(0, len(remaining), batch_size):
            await asyncio.gather(*[recover_child(c) for c in remaining[i:i+batch_size]])

    return total_recovered

async def main():
    SHARDS_DIR = "baukultur_vpr/data/shards"

    # Load the verified extraction list from our persistent log-dump
    with open("corrupted_apple_cells.txt", "r") as f:
        target_cells = [line.strip() for line in f if line.strip()]

    total_cells = len(target_cells)
    logger.info(f"Starting targeted recovery of Apple Lookaround images for {total_cells} sequentially verified cells...")

    writer = wds.ShardWriter(
        os.path.join(SHARDS_DIR, "dataset-apple-recovery-%06d.tar"),
        maxsize=1e9,
        maxcount=5000
    )

    auth = lookaround.Authenticator()
    # Match the main pipeline's concurrency: high semaphore, many concurrent cells
    dl_sem = asyncio.Semaphore(200)
    executor = ProcessPoolExecutor(max_workers=10)
    write_lock = asyncio.Lock()

    overall_recovered = 0

    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0)) as session:
        active_tasks = {}
        max_concurrent_cells = 16  # Process 16 cells concurrently

        for idx, cid in enumerate(target_cells):
            if len(active_tasks) >= max_concurrent_cells:
                done, _ = await asyncio.wait(set(active_tasks.keys()), return_when=asyncio.FIRST_COMPLETED)
                for t in done:
                    recovered = t.result()
                    finished_cid = active_tasks.pop(t)
                    overall_recovered += recovered
                    logger.info(f"Recovered from cell {finished_cid}: {recovered} pairs. (Total: {overall_recovered})")

            task = asyncio.create_task(process_recovery_cell(cid, session, auth, writer, dl_sem, executor, write_lock))
            active_tasks[task] = cid

            if idx % 50 == 0:
                logger.info(f"Feed Progress: {idx}/{total_cells} cells queued. Active: {len(active_tasks)}")

        # Drain remaining tasks
        if active_tasks:
            done, _ = await asyncio.wait(set(active_tasks.keys()))
            for t in done:
                recovered = t.result()
                finished_cid = active_tasks.pop(t)
                overall_recovered += recovered
                logger.info(f"Recovered from cell {finished_cid}: {recovered} pairs. (Total: {overall_recovered})")

    writer.close()
    executor.shutdown()
    logger.info(f"Recovery Complete. Total pairs successfully recovered: {overall_recovered}")

if __name__ == "__main__":
    asyncio.run(main())
