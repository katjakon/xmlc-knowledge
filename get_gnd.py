
import requests
from tqdm import tqdm
from multiprocessing import Pool
import pandas as pd
import json
import random

out = "data/gnd_data.json"
mapping_file = "data/map_idn_with_nid.csv"
mapping = pd.read_csv(mapping_file)
mapping_dict = dict(zip(mapping["idn"], mapping["gnd_id"]))

api_url = "https://lobid.org/gnd/{}.json"

def get_gnd_data(gnd):
    if gnd is None:
        return None
    response = requests.get(api_url.format(gnd))
    if response.status_code == 200:
        return gnd,  response.json()

cores = 6
with open(out, "r") as f:
    gnd_data = json.load(f)
print(f"Number of already mapped data: {len(gnd_data)}")

gnd_idn = list(mapping_dict.values())
print(f"Total Number of GND labes: {len(gnd_idn)}")

gnd_idn = [gnd for gnd in gnd_idn if gnd not in gnd_data]
print(f"Unmapped labels: {len(gnd_idn)}")

gnd_idn = random.choices(gnd_idn, k=10000)

with Pool(processes=cores) as pool:
    for res in pool.imap_unordered(get_gnd_data, gnd_idn):
        if res is not None:
            gnd, result = res
            if gnd not in gnd_data:
                gnd_data[gnd] = result


# Save the data
with open("data/gnd_data.json", "w") as f:
    json.dump(gnd_data, f)