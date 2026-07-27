#!/usr/bin/env python3
"""
脚本功能：根据CSV文件更新JSON配置文件
- 读取CSV文件中的 Target_Species, Tax_ID 信息
- 更新同级目录下的四个JSON文件：
  1. AMP_COMMON_SPECIES.json - 物种别名映射
  2. POSSIBLE_SPECIES.json - 物种属首字母与属名的映射
  3. SPECIES_TAX_ID.json - 物种与Tax_ID的映射
  4. POSSIBLE_GENES.json - 基因名的属首字母与基因种名映射
"""

import json
import time
import os
import sys
import pandas as pd
from collections import defaultdict
from pathlib import Path



def process_csv(csv_file_path:str):
    """
    Args:
        csv_file_path: e.g: "./CAMP3/final_CAMP3_MIC_processed.csv"
    """
    with open ("AMP_COMMON_SPECIES.json","r") as f:
        AMP_COMMON_SPECIES = json.load(f)

    with open ("POSSIBLE_GENES.json","r") as f:
        POSSIBLE_GENES_DATA = json.load(f)
        # Convert list values to sets for efficient adding
        POSSIBLE_GENES = {k: set(v) for k, v in POSSIBLE_GENES_DATA.items()}

    with open ("POSSIBLE_SPECIES.json","r") as f:
        POSSIBLE_SPECIES_DATA = json.load(f)
        # Convert list values to sets for efficient adding
        POSSIBLE_SPECIES = {k: set(v) for k, v in POSSIBLE_SPECIES_DATA.items()}

    with open ("SPECIES_TAX_ID.json","r") as f:
        SPECIES_TAX_ID = json.load(f)


    print(f"Processing CSV: {csv_file_path}")
    df_csv = pd.read_csv(csv_file_path,dtype={"Tax_ID":str})
    df = df_csv[["Tax_ID", "Target_Species"]].copy()
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    df["initial"] = df["Target_Species"].apply(lambda x : x[0])
    df["abbr"] = df["Target_Species"].apply(lambda x: x[0] + ". " + x.split()[1])

    AMP_COMMON_SPECIES_2 = {}
    SPECIES_TAX_ID_2 = {}
    for _, row in df.iterrows():
        AMP_COMMON_SPECIES_2[row['abbr']] = row['Target_Species']
        SPECIES_TAX_ID_2[row['Target_Species']] = row['Tax_ID']
        if row['initial'] not in POSSIBLE_GENES:
            POSSIBLE_GENES[row['initial']] = set()
        if row['initial'] not in POSSIBLE_SPECIES:
            POSSIBLE_SPECIES[row['initial']] = set()
        POSSIBLE_GENES[row['initial']].add(row['Target_Species'].split()[1])
        POSSIBLE_SPECIES[row['initial']].add(row['Target_Species'].split()[0])

    print(f"Before updating AMP_COMMON_SPECIES has {len(AMP_COMMON_SPECIES.values())} species.")
    print(f"Before updating SPECIES_TAX_ID has {len(SPECIES_TAX_ID.values())} species.")
    
    AMP_COMMON_SPECIES.update(AMP_COMMON_SPECIES_2)
    SPECIES_TAX_ID.update(SPECIES_TAX_ID_2)
    print(f"After updating AMP_COMMON_SPECIES has {len(AMP_COMMON_SPECIES.values())} species.")
    print(f"After updating SPECIES_TAX_ID has {len(SPECIES_TAX_ID.values())} species.")
    
    jsons = [AMP_COMMON_SPECIES, SPECIES_TAX_ID, POSSIBLE_GENES, POSSIBLE_SPECIES]
    json_names = ["AMP_COMMON_SPECIES.json",
                  "SPECIES_TAX_ID.json",
                  "POSSIBLE_GENES.json",
                  "POSSIBLE_SPECIES.json"
    ]
    print("\nSaving updated JSON files...")
    for josn_file, json_name in zip(jsons, json_names):
        # Convert sets back to lists for JSON serialization
        if json_name in ["POSSIBLE_GENES.json", "POSSIBLE_SPECIES.json"]:
            josn_file = {k: sorted(list(v)) for k, v in josn_file.items()}
        with open(json_name, "w") as f:
            json.dump(josn_file, f, indent=4)
    print("\n✓ All JSON files updated successfully!")
    

def main():    
    csv_files = [
                # "./APD6/APD6_MIC_final.csv", 
                # "./CAMP3/final_CAMP3_MIC_processed.csv", 
                # "./BaAMP/final_BaAMP_MIC_processed.csv", 
                # "./DBAASP/final_DBAASP_MIC_processed.csv"
                "./grampa/final_GRAMPA_MIC_processed.csv"
                 ]
    for file in csv_files:
        process_csv(file)
        time.sleep(0.5)


    print("All done!")


if __name__ == '__main__':
    main()
