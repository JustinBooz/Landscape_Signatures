/**
 * Swiss Landscape Signatures — DINOv3 7B Interactive Map
 * ======================================================
 * Renders 3.2M points via Deck.gl ScatterplotLayer using binary
 * ArrayBuffer data for zero-parse overhead.
 */

const { DeckGL, ScatterplotLayer, GeoJsonLayer } = deck;

// ---- Configuration ----
const API_URL = '';  // Same origin (served by FastAPI)
const FIELDS_PER_POINT = 9;
const UMAP_SCALE = 0.5;

const GEO_VIEW = {
    longitude: 8.2275,
    latitude: 46.8182,
    zoom: 7.5,
    minZoom: 5,
    maxZoom: 16,
    pitch: 0,
    bearing: 0
};

const UMAP_VIEW = {
    longitude: 0,
    latitude: 0,
    zoom: 4,
    minZoom: 2,
    maxZoom: 14,
    pitch: 0,
    bearing: 0
};

// ---- Global state ----
let pointData = null;       // Float32Array: raw binary
let shardNames = [];        // Array of tar filenames
let shardIndices = null;    // Uint16Array: per-point shard index
let totalPoints = 0;

let activeSpace = 'geo';
let activeColor = 'meso';

// Column offsets
const COL = { lat: 0, lon: 1, umap_x: 2, umap_y: 3, micro: 4, meso: 5, macro: 6, norm_e: 7, norm_n: 8 };
const COLOR_COL = { micro: COL.micro, meso: COL.meso, macro: COL.macro };

// ---- Deck.gl setup ----
const deckgl = new DeckGL({
    container: 'map',
    initialViewState: GEO_VIEW,
    controller: true,
    getTooltip: ({ index }) => {
        if (index < 0 || !pointData) return null;
        const base = index * FIELDS_PER_POINT;
        const cluster = pointData[base + COLOR_COL[activeColor]];
        return cluster === -1 ? 'Noise / Unclustered' : `Cluster ${Math.round(cluster)}`;
    }
});

// Color mapping
function getClusterColor(clusterId) {
    if (clusterId === -1) return [40, 40, 40, 35];
    const hue = (clusterId * 137.508) % 360;
    const s = 0.78, l = 0.58;
    const c = hue / 360;
    const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
    const p = 2 * l - q;
    function h2r(p, q, t) {
        if (t < 0) t += 1;
        if (t > 1) t -= 1;
        if (t < 1/6) return p + (q - p) * 6 * t;
        if (t < 1/2) return q;
        if (t < 2/3) return p + (q - p) * (2/3 - t) * 6;
        return p;
    }
    return [
        Math.round(h2r(p, q, c + 1/3) * 255),
        Math.round(h2r(p, q, c) * 255),
        Math.round(h2r(p, q, c - 1/3) * 255),
        190
    ];
}

// ---- Update layers ----
function updateLayer() {
    if (!pointData) return;
    
    const layers = [];
    
    layers.push(
        new ScatterplotLayer({
            id: 'signatures-points',
            data: { length: totalPoints },
            getPosition: (_, { index }) => {
                const base = index * FIELDS_PER_POINT;
                if (activeSpace === 'geo') {
                    return [pointData[base + COL.lon], pointData[base + COL.lat]];
                } else {
                    return [
                        pointData[base + COL.umap_x] * UMAP_SCALE,
                        pointData[base + COL.umap_y] * UMAP_SCALE
                    ];
                }
            },
            getFillColor: (_, { index }) => {
                const base = index * FIELDS_PER_POINT;
                const cluster = Math.round(pointData[base + COLOR_COL[activeColor]]);
                return getClusterColor(cluster);
            },
            getRadius: (_, { index }) => {
                const base = index * FIELDS_PER_POINT;
                const cluster = pointData[base + COLOR_COL[activeColor]];
                if (cluster === -1) return activeSpace === 'geo' ? 25 : 150;
                return activeSpace === 'geo' ? 50 : 300;
            },
            radiusUnits: 'meters',
            radiusMinPixels: 1,
            radiusMaxPixels: 8,
            pickable: true,
            autoHighlight: true,
            highlightColor: [255, 255, 255, 255],
            onClick: handleClick,
            updateTriggers: {
                getPosition: [activeSpace],
                getFillColor: [activeColor],
                getRadius: [activeColor, activeSpace]
            },
            transitions: {
                getPosition: 1500
            }
        })
    );

    if (activeSpace === 'geo') {
        layers.push(
            new GeoJsonLayer({
                id: 'swiss-border',
                data: `${API_URL}/data/ch_border_4326.geojson`,
                stroked: true,
                filled: false,
                lineWidthMinPixels: 0.5,
                getLineColor: [255, 255, 255, 255],
                pickable: false
            })
        );
    }
    
    deckgl.setProps({ layers });
    updateStats();
}

// ---- Stats ----
function updateStats() {
    if (!pointData) return;
    
    const colIdx = COLOR_COL[activeColor];
    let clusterSet = new Set();
    let noiseCount = 0;
    
    for (let i = 0; i < totalPoints; i++) {
        const cluster = Math.round(pointData[i * FIELDS_PER_POINT + colIdx]);
        if (cluster === -1) noiseCount++;
        else clusterSet.add(cluster);
    }
    
    document.getElementById('stat-points').textContent = totalPoints.toLocaleString();
    document.getElementById('stat-clusters').textContent = clusterSet.size.toLocaleString();
    document.getElementById('stat-noise').textContent = ((noiseCount / totalPoints) * 100).toFixed(1) + '%';
    document.getElementById('stat-visible').textContent = (totalPoints - noiseCount).toLocaleString();
}

// ---- Click handler ----
function handleClick(info) {
    const { index } = info;
    if (index < 0 || !shardIndices || !shardNames.length) return;
    
    const base = index * FIELDS_PER_POINT;
    const lat = pointData[base + COL.lat].toFixed(6);
    const lon = pointData[base + COL.lon].toFixed(6);
    const clusterM = Math.round(pointData[base + COL.micro]);
    const clusterS = Math.round(pointData[base + COL.meso]);
    const clusterL = Math.round(pointData[base + COL.macro]);
    
    const shardIdx = shardIndices[index];
    const tarFile = shardNames[shardIdx] || 'unknown';
    
    const hint = document.getElementById('inspector-hint');
    const img = document.getElementById('inspected-img');
    const loader = document.getElementById('loader');
    const meta = document.getElementById('meta-display');
    
    meta.innerHTML = `
        <b>Point #${index.toLocaleString()}</b><br>
        📍 ${lat}, ${lon}<br>
        🗂 ${tarFile}<br>
        Micro: <b>${clusterM}</b> · Meso: <b>${clusterS}</b> · Macro: <b>${clusterL}</b>
    `;
    
    hint.classList.add('hidden');
    img.style.display = 'none';
    loader.classList.remove('hidden');
    
    fetch(`${API_URL}/closest_image?lat=${lat}&lon=${lon}`)
        .then(resp => resp.json())
        .then(data => {
            const imgUrl = `${API_URL}/image/${data.tar_file}/${data.image_id}`;
            const imageObj = new Image();
            imageObj.onload = () => {
                img.src = imgUrl;
                loader.classList.add('hidden');
                img.style.display = 'block';
            };
            imageObj.onerror = () => {
                loader.classList.add('hidden');
                img.style.display = 'none';
                meta.innerHTML += `<br><span style="color:red">Image not available</span>`;
            };
            imageObj.src = imgUrl;
        })
        .catch(() => {
            loader.classList.add('hidden');
            img.style.display = 'none';
        });
}

// ---- Data loading ----
async function loadData() {
    const progress = document.getElementById('loading-progress');
    
    try {
        progress.textContent = 'Fetching configuration…';
        // Fetch shard names first to verify basic connectivity
        const namesResp = await fetch(`${API_URL}/data/shard_names.json`);
        if (!namesResp.ok) throw new Error(`Server returned ${namesResp.status} for shard_names.json`);
        shardNames = await namesResp.json();

        progress.textContent = 'Fetching point data (115 MB)…';
        const pointsResp = await fetch(`${API_URL}/data/points.bin`);
        if (!pointsResp.ok) throw new Error(`Server returned ${pointsResp.status} for points.bin`);
        const pointsBuffer = await pointsResp.arrayBuffer();
        
        // Safety check for Float32Array multiple of 4
        if (pointsBuffer.byteLength % 4 !== 0) {
            throw new Error(`Data corruption: points.bin size (${pointsBuffer.byteLength}) is not a multiple of 4.`);
        }
        
        pointData = new Float32Array(pointsBuffer);
        totalPoints = pointData.length / FIELDS_PER_POINT;
        
        progress.textContent = `Loaded ${totalPoints.toLocaleString()} points. Fetching indices…`;
        
        const indicesResp = await fetch(`${API_URL}/data/shard_indices.bin`);
        if (!indicesResp.ok) throw new Error(`Server returned ${indicesResp.status} for shard_indices.bin`);
        const indicesBuffer = await indicesResp.arrayBuffer();
        shardIndices = new Uint16Array(indicesBuffer);
        
        progress.textContent = 'Rendering…';
        updateLayer();
        
        setTimeout(() => {
            const overlay = document.getElementById('loading-overlay');
            overlay.classList.add('fade-out');
            setTimeout(() => overlay.remove(), 600);
        }, 300);
        
    } catch (err) {
        progress.textContent = `Error: ${err.message}`;
        console.error('Data loading failed:', err);
        // Show the error in the UI clearly
        const loader = document.querySelector('.loading-spinner');
        if (loader) loader.style.borderTopColor = 'red';
    }
}

// ---- Event listeners ----
document.getElementById('space-select').addEventListener('change', (e) => {
    activeSpace = e.target.value;
    updateLayer();
    deckgl.setProps({
        initialViewState: {
            ...(activeSpace === 'geo' ? GEO_VIEW : UMAP_VIEW),
            transitionDuration: 1500
        }
    });
});

document.getElementById('color-select').addEventListener('change', (e) => {
    activeColor = e.target.value;
    updateLayer();
});

// ---- Start ----
loadData();
