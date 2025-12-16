import pandas as pd
# https://www.kaggle.com/datasets/yorkyong/singapore-passenger-volume-by-train-stations?select=transport_node_train_202308.csv
import numpy as np
import torch

NS = [
    "NS1 Jurong East",
    "NS2 Bukit Batok",
    "NS3 Bukit Gombak",
    "NS4 Choa Chu Kang",
    "NS5 Yew Tee",
    "NS7 Kranji",
    "NS8 Marsiling",
    "NS9 Woodlands",
    "NS10 Admiralty",
    "NS11 Sembawang",
    "NS12 Canberra",
    "NS13 Yishun",
    "NS14 Khatib",
    "NS15 Yio Chu Kang",
    "NS16 Ang Mo Kio",
    "NS17 Bishan",
    "NS18 Braddell",
    "NS19 Toa Payoh",
    "NS20 Novena",
    "NS21 Newton",
    "NS22 Orchard",
    "NS23 Somerset",
    "NS24 Dhoby Ghaut",
    "NS25 City Hall",
    "NS26 Raffles Place",
    "NS27 Marina Bay",
    "NS28 Marina South Pier"
]


EW = [
    "EW1 Pasir Ris",
    "EW2 Tampines",
    "EW3 Simei",
    "EW4 Tanah Merah",
    "EW5 Bedok",
    "EW6 Kembangan",
    "EW7 Eunos",
    "EW8 Paya Lebar",
    "EW9 Aljunied",
    "EW10 Kallang",
    "EW11 Lavender",
    "EW12 Bugis",
    "EW13 City Hall",
    "EW14 Raffles Place",
    "EW15 Tanjong Pagar",
    "EW16 Outram Park",
    "EW17 Tiong Bahru",
    "EW18 Redhill",
    "EW19 Queenstown",
    "EW20 Commonwealth",
    "EW21 Buona Vista",
    "EW22 Dover",
    "EW23 Clementi",
    "EW24 Jurong East",
    "EW25 Chinese Garden",
    "EW26 Lakeside",
    "EW27 Boon Lay",
    "EW28 Pioneer",
    "EW29 Joo Koon",
    "EW30 Gul Circle",
    "EW31 Tuas Crescent",
    "EW32 Tuas West Road",
    "EW33 Tuas Link"
]


NE = [
    "NE1 HarbourFront",
    "NE3 Outram Park",
    "NE4 Chinatown",
    "NE5 Clarke Quay",
    "NE6 Dhoby Ghaut",
    "NE7 Little India",
    "NE8 Farrer Park",
    "NE9 Boon Keng",
    "NE10 Potong Pasir",
    "NE11 Woodleigh",
    "NE12 Serangoon",
    "NE13 Kovan",
    "NE14 Hougang",
    "NE15 Buangkok",
    "NE16 Sengkang",
    "NE17 Punggol"
]


CC = [
    "CC1 Dhoby Ghaut",
    "CC2 Bras Basah",
    "CC3 Esplanade",
    "CC4 Promenade",
    "CC5 Nicoll Highway",
    "CC6 Stadium",
    "CC7 Mountbatten",
    "CC8 Dakota",
    "CC9 Paya Lebar",
    "CC10 MacPherson",
    "CC11 Tai Seng",
    "CC12 Bartley",
    "CC13 Serangoon",
    "CC14 Lorong Chuan",
    "CC15 Bishan",
    "CC16 Marymount",
    "CC17 Caldecott",
    "CC19 Botanic Gardens",
    "CC20 Farrer Road",
    "CC21 Holland Village",
    "CC22 Buona Vista",
    "CC23 one-north",
    "CC24 Kent Ridge",
    "CC25 Haw Par Villa",
    "CC26 Pasir Panjang",
    "CC27 Labrador Park",
    "CC28 Telok Blangah",
    "CC29 HarbourFront"
]


DT = [
    "DT1 Bukit Panjang",
    "DT2 Cashew",
    "DT3 Hillview",
    "DT5 Beauty World",
    "DT6 King Albert Park",
    "DT7 Sixth Avenue",
    "DT8 Tan Kah Kee",
    "DT9 Botanic Gardens",
    "DT10 Stevens",
    "DT11 Newton",
    "DT12 Little India",
    "DT13 Rochor",
    "DT14 Bugis",
    "DT15 Promenade",
    "DT16 Bayfront",
    "DT17 Downtown",
    "DT18 Telok Ayer",
    "DT19 Chinatown",
    "DT20 Fort Canning",
    "DT21 Bencoolen",
    "DT22 Jalan Besar",
    "DT23 Bendemeer",
    "DT24 Geylang Bahru",
    "DT25 Mattar",
    "DT26 MacPherson",
    "DT27 Ubi",
    "DT28 Kaki Bukit",
    "DT29 Bedok North",
    "DT30 Bedok Reservoir",
    "DT31 Tampines West",
    "DT32 Tampines",
    "DT33 Tampines East",
    "DT34 Upper Changi",
    "DT35 Expo"
]


TE = [
    "TE1 Woodlands North",
    "TE2 Woodlands",
    "TE3 Woodlands South",
    "TE4 Springleaf",
    "TE5 Lentor",
    "TE6 Mayflower",
    "TE7 Bright Hill",
    "TE8 Upper Thomson",
    "TE9 Caldecott",
    "TE11 Stevens",
    "TE12 Napier",
    "TE13 Orchard Boulevard",
    "TE14 Orchard",
    "TE15 Great World",
    "TE16 Havelock",
    "TE17 Outram Park",
    "TE18 Maxwell",
    "TE19 Shenton Way",
    "TE20 Marina Bay",
]


BP = [
    "BP1 Bukit Panjang",
    "BP2 South View",
    "BP3 Keat Hong",
    "BP4 Teck Whye",
    "BP5 Phoenix",
    "BP6 Bukit Panjang",
    "BP7 ",
    "BP8 "
]


SK = [
    "STC Sengkang",
    "SW1 Cheng Lim",
    "SW2 Farmway",
    "SW3 Kupang",
    "SW4 Thanggam",
    "SW5 Fernvale",
    "SW6 Layar",
    "SW7 Tongkang",
    "SW8 Renjong",
    "SE1 Compassvale",
    "SE2 Rumbia",
    "SE3 Bakau",
    "SE4 Kangkar",
    "SE5 Ranggung"
]


PG = [
    "PTC Punggol",
    "PW1 Sam Kee",
    "PW3 Punggol Point",
    "PW4 Teck Lee",
    "PW5 Samudera",
    "PW6 Nibong",
    "PW7 Sumang",
    "PE1 Cove",
    "PE2 Meridian",
    "PE3 Coral Edge",
    "PE4 Riviera",
    "PE5 Kadaloor",
    "PE6 Oasis",
    "PE7 Damai"
]

data = pd.read_csv("../cleaning_scripts/station_id_mapping.csv").set_index("Station_Code").to_dict(orient = "index")


def get_edge(li):
    new_li = [data[i.split(" ")[0]]["Station_ID"] for i in li]
    return new_li

def get_NS():
    return get_edge(NS)

def get_EW():
    return get_edge(EW)

def get_NE():
    return get_edge(NE)

def get_CC():
    return get_edge(CC)

def get_DT():
    return get_edge(DT)

def get_TE():
    return get_edge(TE)

def get_BP():
    return get_edge(BP)

def get_SK():
    return get_edge(SK)

def get_PG():
    return get_edge(PG)

def main():
    print(get_PG())

if __name__ == "__main__":
    main()


node_data = pd.read_csv("../cleaning_scripts/combined_node_features.csv")


stations = sorted(node_data["STATION_ID"].unique())
def get_stations():
    return stations
N = len(stations)
print(N)
compiled_data = []
for m in node_data["MONTH"].unique():
    for w in node_data["IS_WEEKEND"].unique():
        for t in node_data["TIME_PER_HOUR"].unique():
            curr = node_data[(node_data["MONTH"] == m) & (node_data["TIME_PER_HOUR"] == t) & (node_data["IS_WEEKEND"] == w)][["STATION_ID", "MONTH", "TIME_PER_HOUR", "IN_FLOW_NORM", "OUT_FLOW_NORM", "IS_WEEKEND"]]
            curr = curr.set_index("STATION_ID").reindex(stations).fillna(0.0)
            curr["sin_time"] = np.sin(2*np.pi*t/24.0)
            curr["cos_time"] = np.cos(2*np.pi*t/24.0)
            curr = curr[["IN_FLOW_NORM", "OUT_FLOW_NORM", "IS_WEEKEND", "MONTH", "sin_time", "cos_time"]]
            compiled_data.append(curr.values.tolist()) # for every t
print(curr.columns.tolist())

compiled_data = torch.tensor(compiled_data, dtype = torch.float32)

def return_data():
    return compiled_data

od = pd.read_csv("../cleaning_scripts/combined_train_flow_mapped.csv")
od_uv = od.groupby(["ORIGIN_ID","DEST_ID"])["TOTAL_TRIPS"].sum()

def edge_weights(edge_index):
    E = edge_index.size(1)
    w = torch.zeros(E, dtype=torch.float32)
    for e in range(E):
        u = int(edge_index[0, e])
        v = int(edge_index[1, e])
        w[e] = od_uv.get((u, v), 0.0)   # 0 if no OD observed
    w = torch.log1p(w)
    if w.max() > 0:
        w = w / w.max()
    edge_weight = 0.5 + 0.5 * w 
    return edge_weight
