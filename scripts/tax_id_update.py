# 更新SPECIES_TAX_ID.json的脚本 - 支持多线程并行处理
import pandas as pd
import json
from Bio import Entrez
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from typing import Tuple, Dict
import time

Entrez.email = "luyeqing21@mail.ustc.edu.cn"


def NCBI_search(species:str, genus:str)->Tuple[str,str]:
    """
    搜索NCBI分类数据库获取物种的Taxonomy ID和Scientific Name
    
    :param species: 物种名首字母 (e.g., 'S')
    :param genus: 属名 (e.g., 'aureus')
    :return: (tax_id, scientific_name) 或 (None, None)
    """
    term = f"{species} {genus}[Scientific Name]"
    
    try:
        # 搜索阶段
        try:
            handle = Entrez.esearch(db="taxonomy", term=term, retmax=1)
            record = Entrez.read(handle)
            handle.close()
        except Exception as e:
            print(f"NCBI搜索失败 (species={species}, genus={genus}): {type(e).__name__}: {str(e)}")
            return None, None
        
        # 验证搜索结果
        if not record.get("IdList"):
            print(f"未找到物种 {species}. {genus} 的Taxonomy ID")
            return None, None
        
        # 获取第一个结果的ID
        tax_id = record["IdList"][0]
        
        # 获取详细信息阶段
        try:
            summary = Entrez.efetch(db="taxonomy", id=tax_id, retmode="xml")
            data = Entrez.read(summary)
            summary.close()
        except Exception as e:
            print(f"获取物种(species={species}, genus={genus})Taxonomy详情失败 (tax_id={tax_id}): {type(e).__name__}: {str(e)}")
            return None, None
        
        # 验证返回数据结构
        if not data or len(data) == 0:
            print(f"Taxonomy数据为空 (tax_id={tax_id})")
            return None, None
        
        if "ScientificName" not in data[0]:
            print(f"Taxonomy数据缺少ScientificName字段 (tax_id={tax_id})")
            return None, None
        
        sci_name = data[0]["ScientificName"]
        return tax_id, sci_name
        
    except Exception as e:
        print(f"NCBI_search发生未预期的错误 (species={species}, genus={genus}): {type(e).__name__}: {str(e)}")
        return None, None


# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 线程池配置
MAX_WORKERS = 5  # 最大线程数（NCBI API限制）

def _search_species_tax_id(species_full_name: str) -> Tuple[str, str, str]:
    """
    辅助函数：搜索单个物种的Tax_ID（用于多线程）
    :param species_full_name: 物种全名 (e.g., "Staphylococcus aureus")
    :return: (original_name, scientific_name, tax_id) 或 (original_name, None, None)
    """
    try:
        parts = species_full_name.split()
        if len(parts) < 2:
            logger.warning(f"物种名格式不正确: {species_full_name}")
            return species_full_name, None, None
        
        species, genus = parts[0], parts[1]
        tax_id, sci_name = NCBI_search(species, genus)
        
        if tax_id and sci_name:
            logger.info(f"✓ 已获取: {species_full_name} -> Tax_ID: {tax_id}, Scientific_Name: {sci_name}")
            return species_full_name, sci_name, tax_id
        else:
            logger.warning(f"✗ 未找到: {species_full_name}")
            return species_full_name, None, None
    except Exception as e:
        logger.error(f"处理物种失败 ({species_full_name}): {e}")
        return species_full_name, None, None


def update_species_tax_id_parallel(species_list: list, existing_taxid: dict, workers: int = MAX_WORKERS) -> Dict[str, str]:
    """
    并行更新物种的Tax_ID
    :param species_list: 需要查询的物种列表
    :param existing_taxid: 已有的物种Tax_ID字典
    :param workers: 线程数
    :return: 更新后的物种Tax_ID字典
    """
    # 筛选出需要更新的物种
    species_to_update = [sp for sp in species_list if sp not in existing_taxid.keys()]
    
    logger.info(f"共需要更新 {len(species_to_update)} 个物种 (总计 {len(species_list)} 个，已有 {len(existing_taxid)} 个)")
    
    if not species_to_update:
        logger.info("所有物种已存在，无需更新")
        return existing_taxid
    
    updated_count = 0
    failed_count = 0
    
    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_search_species_tax_id, sp): sp for sp in species_to_update}
            
            for i, future in enumerate(as_completed(futures), 1):
                try:
                    orig_name, sci_name, tax_id = future.result(timeout=30)
                    if tax_id and sci_name:
                        existing_taxid[sci_name] = str(tax_id)
                        updated_count += 1
                    else:
                        failed_count += 1
                    
                    if i % 10 == 0:
                        logger.info(f"进度: {i}/{len(species_to_update)} (成功: {updated_count}, 失败: {failed_count})")
                except Exception as e:
                    sp = futures[future]
                    logger.error(f"处理任务异常 ({sp}): {e}")
                    failed_count += 1
    except Exception as e:
        logger.error(f"多线程处理异常: {e}")
        # 回退到顺序处理
        logger.info("回退到顺序处理模式...")
        for sp in species_to_update:
            orig_name, sci_name, tax_id = _search_species_tax_id(sp)
            if tax_id and sci_name:
                existing_taxid[sci_name] = str(tax_id)
                updated_count += 1
            else:
                failed_count += 1
            time.sleep(0.5)  # 避免请求过频繁
    
    logger.info(f"更新完成: 成功 {updated_count} 个，失败 {failed_count} 个")
    return existing_taxid

df = pd.read_csv('./APD6/APD6_MIC_final.csv')
species_list = df['Target_Species'].unique().tolist()

with open('SPECIES_TAX_ID.json', 'r') as f:
    species_taxid = json.load(f)

logger.info(f"从CSV读取 {len(species_list)} 个物种")
logger.info(f"从JSON读取 {len(species_taxid)} 个已有的物种Tax_ID映射")

# 使用多线程更新物种Tax_ID
species_taxid = update_species_tax_id_parallel(species_list, species_taxid, workers=MAX_WORKERS)

# 保存更新后的数据
with open('SPECIES_TAX_ID.json', 'w') as f:
    json.dump(species_taxid, f, indent=4, ensure_ascii=False)

logger.info(f"已保存更新后的SPECIES_TAX_ID.json，共包含 {len(species_taxid)} 个物种")