import geopandas as gpd
import os

gdb_path = "/home/jubooz/landscape_signatures/geodata/LABES/labes_36_zersiedlung/LABES_36_GIS/LABES_36_Zersiedlung_2018.gdb"
try:
    layers = gpd.list_layers(gdb_path)
    print(f"Layers in {os.path.basename(gdb_path)}:")
    print(layers)
except Exception as e:
    print(f"Error listing layers: {e}")
