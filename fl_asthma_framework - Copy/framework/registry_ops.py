import json
import os

REG_PATH = "data/hospitals.json"

def load_registry(path=REG_PATH):
    with open(path, "r") as f:
        return json.load(f)

def save_registry(reg, path=REG_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(reg, f, indent=2)

def add_hospital(reg, hospital_id, name, csv_path):
    # Prevent duplicates
    for h in reg["hospitals"]:
        if h["id"] == hospital_id:
            raise ValueError(f"Hospital id '{hospital_id}' already exists.")
    reg["hospitals"].append({"id": hospital_id, "name": name, "path": csv_path})
    return reg

def remove_hospital(reg, hospital_id):
    reg["hospitals"] = [h for h in reg["hospitals"] if h["id"] != hospital_id]
    return reg
