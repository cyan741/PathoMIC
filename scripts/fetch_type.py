import json
import time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
import bacdive
from ete3 import NCBITaxa
import logging
from typing import Tuple, Dict
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
MAX_WORKERS = 5  # 最大线程数（避免请求过快）
REQUEST_DELAY = 0.5  # 请求之间的延迟（秒）
SPECIES_TAX_ID_FILE = "SPECIES_TAX_ID.json"
OUTPUT_CSV_FILE = "SPECIES_TYPE.csv"
EMAIL = 'luyeqing21@mail.ustc.edu.cn'
PASSWORD = 'BacDive@lyq741'

def read_species_data(file_path: str) -> Dict:
    """
    从JSON文件读取物种数据
    
    Args:
        file_path: JSON文件路径
        
    Returns:
        物种数据字典
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logging.info(f"成功读取 {file_path}，包含 {len(data)} 个物种")
        return data
    except Exception as e:
        logging.error(f"读取文件 {file_path} 失败: {e}")
        return {}

def is_fungi_by_taxonomy(tax_id: str) -> bool:
    """
    判断是否为真菌
    """
    try:
        ncbi = NCBITaxa()
        tax_id = int(tax_id)
        lineage = ncbi.get_lineage(tax_id)
        names = ncbi.get_taxid_translator(lineage)
        ranks = ncbi.get_rank(lineage)

        for tid in lineage:
            if ranks.get(tid) == "kingdom" and names[tid] == "Fungi":
                return True
        return False
    except Exception as e:
        logging.error(f"判断真菌失败 (tax_id={tax_id}): {e}")
        return False

def get_gram_from_bacdive(scientific_name: str) -> str:
    """
    从 BacDive 获取革兰氏染色信息
    返回: Gram-positive / Gram-negative / None
    """
    try:
        # 在当前线程中创建新的 BacDive 客户端（每个线程有自己的 SQLite 连接）
        client = bacdive.BacdiveClient(EMAIL, PASSWORD)
        client.setSearchType('exact')
        
        count = client.search(taxonomy=scientific_name)

        if count == 0:
            logging.warning(f"BacDive 中未找到: {scientific_name}")
            return None

        gram = None
        # 遍历所有匹配的 strain
        for strain in client.retrieve():
            try:
                morphology = strain.get("Morphology", {})
                if not morphology:
                    continue
                cell_morphology = morphology.get("cell morphology", [])
                if not cell_morphology:
                    continue
                if isinstance(cell_morphology, list):
                    cell_morphology = cell_morphology[0]
                gram = cell_morphology.get("gram stain", None)
                if gram == 'negative' or gram == 'positive':
                    break
            except Exception as e:
                logging.debug(f"处理 strain 数据时出错: {e}")
                continue

        if gram == 'negative':
            return 'Gram-negative'
        elif gram == 'positive':
            return 'Gram-positive'
        else:
            return None

    except Exception as e:
        logging.error(f"从 BacDive 获取信息失败 ({scientific_name}): {e}")
        return None

def get_type_thread_safe(tax_id: str, sci_name: str) -> str:
    """
    线程安全的获取物种类型
    返回: Gram-positive / Gram-negative / Fungi / unknown
    """
    try:
        if is_fungi_by_taxonomy(tax_id):
            return 'Fungi'
        else:
            gram = get_gram_from_bacdive(sci_name)
            return gram if gram else "unknown"
    except Exception as e:
        logging.error(f"获取类型失败 ({sci_name}, tax_id={tax_id}): {e}")
        return "unknown"

def process_single_species(species_name: str, tax_id: str) -> Tuple[str, str, str, str]:
    """
    处理单个物种，获取其类型（线程安全）
    
    Args:
        species_name: 物种学名
        tax_id: Taxonomy ID
        
    Returns:
        (species_name, tax_id, type, status)
    """
    try:
        # 获取物种类型（每个线程独立获取）
        microbe_type = get_type_thread_safe(str(tax_id), species_name)
        
        if microbe_type is None:
            microbe_type = "unknown"
        
        logging.info(f"处理完成: {species_name} (tax_id={tax_id}) -> Type: {microbe_type}")
        return species_name, tax_id, microbe_type, "success"
        
    except Exception as e:
        logging.error(f"处理物种 {species_name} (tax_id={tax_id}) 时出错: {e}")
        return species_name, tax_id, "unknown", "failed"

def process_species_parallel(species_data: Dict, max_workers: int = 3) -> Dict:
    """
    使用多线程并行处理物种（每个线程独立创建数据库连接）
    
    Args:
        species_data: 物种数据字典 {species_name: {"TAX_ID": "xxx", "Type": "xxx"}}
        max_workers: 最大线程数
        
    Returns:
        处理结果字典
    """
    results = {}
    
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交任务
            futures = {}
            for species_name, data in species_data.items():
                tax_id = data.get("TAX_ID", "")
                future = executor.submit(process_single_species, species_name, tax_id)
                futures[future] = species_name
            
            # 处理完成的任务
            completed = 0
            total = len(futures)
            
            for future in as_completed(futures):
                try:
                    species_name, tax_id, microbe_type, status = future.result(timeout=60)
                    results[species_name] = {
                        "TAX_ID": tax_id,
                        "Type": microbe_type
                    }
                    completed += 1
                    
                    if completed % 5 == 0:
                        logging.info(f"处理进度: {completed}/{total}")
                    
                    # 在请求之间添加延迟
                    time.sleep(REQUEST_DELAY)
                    
                except Exception as e:
                    species_name = futures[future]
                    logging.error(f"处理任务失败 ({species_name}): {e}")
                    results[species_name] = {
                        "TAX_ID": species_data[species_name].get("TAX_ID", ""),
                        "Type": "unknown"
                    }
        
        logging.info(f"多线程处理完成，共处理 {len(results)} 个物种")
        return results
        
    except Exception as e:
        logging.error(f"多线程处理异常: {e}")
        return {}

def save_to_json(data: Dict, file_path: str):
    """
    保存数据到JSON文件
    
    Args:
        data: 数据字典
        file_path: 输出文件路径
    """
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        logging.info(f"成功保存数据到 {file_path}")
    except Exception as e:
        logging.error(f"保存JSON文件 {file_path} 失败: {e}")

def save_to_csv(data: Dict, file_path: str):
    """
    保存数据到CSV文件
    
    Args:
        data: 数据字典
        file_path: 输出文件路径
    """
    try:
        records = []
        for species_name, info in data.items():
            records.append({
                "Species_Name": species_name,
                "TAX_ID": info.get("TAX_ID", ""),
                "Type": info.get("Type", "unknown")
            })
        
        df = pd.DataFrame(records)
        df.to_csv(file_path, index=False, encoding="utf-8")
        logging.info(f"成功保存数据到 {file_path}，共 {len(df)} 行")
    except Exception as e:
        logging.error(f"保存CSV文件 {file_path} 失败: {e}")
# 主处理流程
def main():
    species_data = read_species_data(SPECIES_TAX_ID_FILE)
    if not species_data:
        logging.error("没有读取到物种数据，程序退出")
        return
    
    # # test
    # species_data = dict(list(species_data.items())[:10])
    # print("测试数据前十个物种")

    # 2. 使用多线程处理物种
    results = process_species_parallel(species_data, max_workers=MAX_WORKERS)
    if not results:
        logging.error("物种处理失败，程序退出")
        return
    
    # 3. 保存结果
    save_to_json(results, SPECIES_TAX_ID_FILE)
    save_to_csv(results, OUTPUT_CSV_FILE)
    
    logging.info("处理完成！")

if __name__ == "__main__":
    main()