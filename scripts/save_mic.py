import json
import re
import os
import pandas as pd
from typing import Dict, List, Tuple
import html 
from Bio.SeqUtils import molecular_weight
import time
from rapidfuzz.distance import Levenshtein
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from Bio import Entrez
import logging
import random
import numpy as np

Entrez.email = "luyeqing21@mail.ustc.edu.cn"
Entrez.tool = "AMPAnalysis"

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 线程池配置
MAX_WORKERS = 5  # 最大线程数（NCBI API限制）
REQUEST_DELAY = 1.0  # 请求延迟（秒）, NCBI建议至少1秒
MAX_RETRIES = 5  # 最大重试次数
RETRY_BACKOFF_BASE = 2  # 指数退避基数（2秒, 4秒, 8秒...）

# 统一一下物种名称第一个是species，第二个是genes， 跟json文件对应起来
with open("./POSSIBLE_GENES.json", "r") as f:
    POSSIBLE_GENES = json.load(f)

with open("./POSSIBLE_SPECIES.json", "r") as f:
    POSSIBLE_SPECIES = json.load(f)

with open("./AMP_COMMON_SPECIES.json", "r") as f:
    AMP_COMMON_SPECIES = json.load(f)

with open("./SPECIES_TAX_ID.json", "r") as f:
    SPECIES_TAX_ID = json.load(f)

def convert_mic_to_log_uM(df: pd.DataFrame, mic_column: str = "MIC_Value(ug/ml)", sequence_column: str = "Sequence") -> pd.DataFrame:
    """
    将MIC_value(ug/ml)转换为log_MIC_Value(uM)
    
    转换过程：
    1. 使用Bio.SeqUtils.molecular_weight计算序列的分子量(g/mol)
    2. 转换公式: uM = (ug/ml) * 1000 / MW
    3. 取对数: log_MIC_Value = log10(uM)
    
    Args:
        df: 包含MIC值和序列的dataframe
        mic_column: MIC值列名，默认"MIC_Value(ug/ml)"
        sequence_column: 序列列名，默认"Sequence"
    
    Returns:
        添加了log_MIC_Value(uM)列的dataframe
    """
    df = df.copy()
    
    def calculate_log_uM(row):
        try:
            mic_ug_ml = float(row[mic_column])
            sequence = row[sequence_column]
            
            if pd.isna(mic_ug_ml) or pd.isna(sequence):
                return np.nan
            
            # 计算分子量 (g/mol)，对于氨基酸序列
            mw = molecular_weight(sequence, seq_type="protein")
            
            if mw <= 0:
                return np.nan
            
            # 转换为uM: (ug/ml) * 1000 / MW(g/mol)
            uM = mic_ug_ml * 1000 / mw
            
            # 取对数
            log_uM = np.log10(uM)
            
            return log_uM
        except Exception as e:
            logger.warning(f"计算log_MIC_Value失败: {e}, sequence: {row.get(sequence_column, 'N/A')}")
            return np.nan
    
    df["log_MIC_Value(uM)"] = df.apply(calculate_log_uM, axis=1)
    
    return df

def extract_bacterium(species_part:str, direct_NCBI_search:bool, abbr:bool = False)->Tuple[str,str]:
    '''
    :param species_part: eg. "A. baumanii"
    :type species_part: str
    :type direct_NCBI_search: bool
    :type abbr: bool
    :return: tax_id, species 
    :rtype: Tuple[str, str]
    '''
    if "." in species_part: # 缩写形式
        tax_id, species = abbr2sci_name(species_part)
    else: # 全称形式
        try:
            tax_id = SPECIES_TAX_ID[species_part]
            return tax_id, species_part
        except KeyError:
            print(f"无法在本地为物种名称找到Taxonomy ID: {species_part}\n")
            sci_name_parts = species_part.split()
            tax_id, species = NCBI_search(sci_name_parts[0], sci_name_parts[1])
            if not tax_id:# 考虑拼写错误
                initial_gene = sci_name_parts[0][0] + ". " + sci_name_parts[1]
                tax_id, species = abbr2sci_name(initial_gene)



    # if not direct_NCBI_search and not abbr:
    #     # 在本地的AMP_COMMON_SPECIES里面找不到全称的直接丢掉
    #     try:
    #         tax_id = SPECIES_TAX_ID[species_part]
    #         return tax_id, species_part
    #     except KeyError:
    #         print(f"无法在本地为物种名称找到Taxonomy ID: {species_part}\n")
    #         sci_name_parts = species_part.split()
    #         tax_id, species = NCBI_search(sci_name_parts[0], sci_name_parts[1])
    #         if not tax_id:# 考虑拼写错误
    #             initial_gene = sci_name_parts[0][0] + ". " + sci_name_parts[1]
    #             tax_id, species = abbr2sci_name(initial_gene)
                
    # if not direct_NCBI_search and abbr:
    #     tax_id, species = abbr2sci_name(species_part)
    # elif abbr:
    #     # 在本地的AMP_COMMON_SPECIES里面找不到全称的直接丢掉
    #     try:
    #         sci_name = AMP_COMMON_SPECIES[species_part]
    #     except KeyError:
    #         print(f"无法为物种名称找到Scientific Name: {species_part}\n")
    #         return None, None
    #     sci_name_parts = sci_name.split()
    #     tax_id, species = NCBI_search(sci_name_parts[0], sci_name_parts[1])
    # if direct_NCBI_search and abbr:
    #     tax_id, species = abbr2sci_name(species_part)
    # elif direct_NCBI_search:
    #     sci_name_parts = species_part.split()
    #     tax_id, species = NCBI_search(sci_name_parts[0], sci_name_parts[1])

    if not tax_id:
        print(f"无法为物种名称找到Taxonomy ID: {species_part}\n")
    return tax_id, species

def correct_species_with_distance(
    species_initial,
    genus_raw, 
    max_distance=2
)->Tuple[str, int, str]:
    """
    raw_name: e.g. 'S. aureu'
    max_distance: 建议 1 或 2（非常重要）
    """

    best_match = None
    best_distance = None
    if species_initial not in POSSIBLE_GENES.keys(): # 如果best_distance是None，说明initial没有对应的基因列表，返回给第三步不用在species里面匹配
        return f"{species_initial}. {genus_raw}", None, "unchanged"
    genus_list = POSSIBLE_GENES[species_initial]

    for genus in genus_list:
        dist = Levenshtein.distance(genus_raw, genus)

        if best_distance is None or dist < best_distance:
            best_distance = dist
            best_match = genus

    if best_distance is not None and best_distance <= max_distance:
        corrected = f"{species_initial}. {best_match}"
        return corrected, best_distance, "corrected"
    raw_name = f"{species_initial}. {genus_raw}"
    return raw_name, best_distance, "unchanged"

def _search_candidate_species(species_genus_pair: Tuple[str, str]) -> Tuple[str, str, str]:
    """
    辅助函数：搜索单个候选物种（用于多线程）
    :return: (sp_name, tax_id, sci_name)
    """
    sp, genus = species_genus_pair
    tax_id, sci_name = NCBI_search(sp, genus)
    return sp, tax_id, sci_name

def NCBI_search(species:str, genus:str)->Tuple[str,str]:
    """
    搜索NCBI分类数据库获取物种的Taxonomy ID和Scientific Name
    带指数退避重试策略，处理429错误
    
    :param species: 物种名首字母 (e.g., 'S')
    :param genus: 属名 (e.g., 'aureus')
    :return: (tax_id, scientific_name) 或 (None, None)
    """
    term = f"{species} {genus}[Scientific Name]"
    
    # 指数退避重试
    for attempt in range(MAX_RETRIES):
        try:
            # 搜索阶段
            try:
                handle = Entrez.esearch(db="taxonomy", term=term, retmax=1)
                record = Entrez.read(handle)
                handle.close()
                # 请求成功后添加延迟
                time.sleep(REQUEST_DELAY + random.uniform(0, 0.5))
            except Exception as e:
                # 检查是否是429错误
                error_str = str(e)
                if "429" in error_str or "Too Many Requests" in error_str:
                    if attempt < MAX_RETRIES - 1:
                        wait_time = RETRY_BACKOFF_BASE ** attempt
                        logger.warning(f"收到429错误，等待{wait_time}秒后重试 (attempt {attempt+1}/{MAX_RETRIES}): {species} {genus}")
                        time.sleep(wait_time)
                        continue
                logger.debug(f"NCBI搜索失败 (species={species}, genus={genus}): {type(e).__name__}: {str(e)}")
                return None, None
            
            # 验证搜索结果
            if not record.get("IdList"):
                logger.debug(f"未找到物种 {species} {genus} 的Taxonomy ID")
                return None, None
            
            # 获取第一个结果的ID
            tax_id = record["IdList"][0]
            
            # 获取详细信息阶段
            try:
                summary = Entrez.efetch(db="taxonomy", id=tax_id, retmode="xml")
                data = Entrez.read(summary)
                summary.close()
                # 请求成功后添加延迟
                time.sleep(REQUEST_DELAY + random.uniform(0, 0.5))
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "Too Many Requests" in error_str:
                    if attempt < MAX_RETRIES - 1:
                        wait_time = RETRY_BACKOFF_BASE ** attempt
                        logger.warning(f"收到429错误，等待{wait_time}秒后重试 (attempt {attempt+1}/{MAX_RETRIES}): {species} {genus}")
                        time.sleep(wait_time)
                        continue
                logger.debug(f"获取物种(species={species}, genus={genus})Taxonomy详情失败 (tax_id={tax_id}): {type(e).__name__}: {str(e)}")
                return None, None
            
            # 验证返回数据结构
            if not data or len(data) == 0:
                logger.debug(f"Taxonomy数据为空 (tax_id={tax_id})")
                return None, None
            
            if "ScientificName" not in data[0]:
                logger.debug(f"Taxonomy数据缺少ScientificName字段 (tax_id={tax_id})")
                return None, None
            
            sci_name = data[0]["ScientificName"]
            return tax_id, sci_name
        
        except Exception as e:
            logger.error(f"NCBI_search发生未预期的错误 (species={species}, genus={genus}): {type(e).__name__}: {str(e)}")
            return None, None
    
    # 所有重试都失败了
    logger.error(f"经过{MAX_RETRIES}次重试，仍无法获取物种 {species} {genus} 的Taxonomy ID")
    return None, None

def abbr2sci_name(abbr_name: str) -> Tuple[str, str]:
    '''    
    :param abbr_name: Description
    :type abbr_name: str
    :return: tax_id, sci_name
    :rtype: Tuple[str, str]
    '''

    abbr_name_set = AMP_COMMON_SPECIES.keys()
    # 缩写可以直接在本地找到
    if abbr_name in abbr_name_set:
        sci_name = AMP_COMMON_SPECIES[abbr_name]
        tax_id = SPECIES_TAX_ID[sci_name]  
        return tax_id, sci_name
    # 如果不在预定义的缩写中，先检查是否种名拼写有问题
    names = abbr_name.split()
    initial = names[0][0]
    genus = names[1]
    corrected_species, distance, status = correct_species_with_distance(initial, genus)
    if status == "corrected":
        sci_name = AMP_COMMON_SPECIES[corrected_species]
        tax_id = SPECIES_TAX_ID[sci_name]  
        return tax_id, sci_name
    
    # 最后尝试在POSSIBLE_SPECIES.json里面使用首字母组合，使用NCBI API搜索
    if not distance and initial not in POSSIBLE_SPECIES.keys():
        logger.warning(f"物种名称: {abbr_name},首字母没有出现在本地")
        return None, None
    
    candidate_species = POSSIBLE_SPECIES[initial]
    
    # 使用多线程并行搜索候选物种
    species_genus_pairs = [(sp, genus) for sp in candidate_species]
    
    try:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(_search_candidate_species, pair): pair for pair in species_genus_pairs}
            
            for future in as_completed(futures):
                try:
                    sp, tax_id, sci_name = future.result(timeout=10)
                    if tax_id:
                        logger.info(f"找到物种: {abbr_name} -> {sci_name} (tax_id={tax_id})")
                        return tax_id, sci_name
                except Exception as e:
                    logger.debug(f"搜索失败: {e}")
                    continue
    except Exception as e:
        logger.error(f"多线程搜索异常: {e}")
        # 回退到顺序搜索
        for sp in candidate_species:
            tax_id, sci_name = NCBI_search(sp, genus)
            if tax_id:
                return tax_id, sci_name
            time.sleep(0.5)

    return None, None

def get_tax_id(species_name: str) -> str:
    """
    获取物种的Taxonomy ID
    带指数退避重试策略，处理429错误
    
    :param species_name: 物种全名
    :return: tax_id 或 None
    """
    term = f"{species_name}[Scientific Name]"
    for attempt in range(MAX_RETRIES):
        try:
            handle = Entrez.esearch(db="taxonomy", term=term, retmax=1)
            record = Entrez.read(handle)
            handle.close()
            # 请求成功后添加延迟
            time.sleep(REQUEST_DELAY + random.uniform(0, 0.5))
            
            if not record.get("IdList"):
                logger.debug(f"未找到物种: {species_name}")
                return None
            tax_id = record["IdList"][0]
            return str(tax_id)
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "Too Many Requests" in error_str:
                if attempt < MAX_RETRIES - 1:
                    wait_time = RETRY_BACKOFF_BASE ** attempt
                    logger.warning(f"收到429错误，等待{wait_time}秒后重试 (attempt {attempt+1}/{MAX_RETRIES}): {species_name}")
                    time.sleep(wait_time)
                    continue
            logger.error(f"获取Tax_ID失败 ({species_name}): {e}")
            return None

def unify_to_ug_ml(row):
    """
    统一MIC值单位为μg/mL
    """
    raw_val = str(row['MIC_Value']).lower()
    seq = row['Sequence']
    
    try:
        # 1. 提取数值
        match = re.search(r'([\d\.]+)', raw_val)
        if not match: 
            return None
        value = float(match.group(1))

        # 2. 如果已经是 ug/ml，直接返回数值
        if 'ug/ml' in raw_val:
            return value
            
        # 3. 如果是 uM，则根据序列计算分子量并转换
        if 'um' in raw_val:
            # 计算分子量 (Da)
            mw = molecular_weight(seq, seq_type="protein")
            # 转换公式：(uM * MW) / 1000 = ug/ml
            return (value * mw) / 1000
        
        return value # 默认返回
    except Exception as e:
        logger.warning(f"MIC值转换失败 ({raw_val}): {e}")
        return None

# 氨基酸序列要清洗，保留20种标准的氨基酸字母
def clean_sequence(seq: str) -> str:
    seq = seq.upper().strip()
    standard_aa = set("ACDEFGHIKLMNPQRSTVWY")
    if set(seq).issubset(standard_aa):
        return seq
    else:
        return None


def _process_raw_species(raw_species: str) -> Tuple[str, str, str]:
    """
    处理单个物种字符串（用于多线程）
    :return: (raw, Tax_ID, Target_Species)
    """
    tax_id, species = extract_bacterium(raw_species, direct_NCBI_search=False, abbr=False)
    return raw_species, tax_id, species

def split_strain_species(raw_species:str)->Tuple[str, str]:
    parts = raw_species.split()
    if "." in parts[0] and parts[0][-1] != ".":
        # E.coli 0111
        species_parts = parts[0].split(".")
        species_part = ". ".join(species_parts)
        strain_part = " ".join(parts[1:])
        return species_part, strain_part
    if len(parts) > 2:
        species_part = " ".join(parts[:2]) # "A. baumanii"
        strain_part = " ".join(parts[2:])   # "CI 2675"
    else:
        species_part = raw_species
        strain_part = ""

    return species_part, strain_part

def extract_raw_species(col_name):
    """从MIC(物种名)格式中提取物种名，如果没有匹配则保留原值"""
    match = re.search(r'MIC\s*\((.*)\)', str(col_name))
    if match:
        return match.group(1).strip()
    else:
        return col_name


def main():
    parser = argparse.ArgumentParser(description="Process APD6 MIC data.")
    parser.add_argument("--input_path", "-I", type=str, default="APD6", help="Input CSV path to process.")
    parser.add_argument("--input_file", "-F", type=str, default="APD6_processed.csv", help="Input CSV file to process.")
    parser.add_argument("--output_path", "-O", type=str, default="APD6", help="Output CSV path name.")
    parser.add_argument("--from_wide_format", "-W", type=bool, default=False, help="Transfer from wide format.")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS, help="Number of worker threads.")
    arg = parser.parse_args()

    FROM_WIDE_FORMAT = arg.from_wide_format
    csv2process = os.path.join(arg.input_path, arg.input_file)
    logger.info(f"开始处理文件: {csv2process}")
    
    if FROM_WIDE_FORMAT:
        # 读取原始csv文件并清洗氨基酸序列
        logger.info(f"从宽表转换为长表")
        df = pd.read_csv(csv2process, dtype={"ID": str})
        logger.info(f"初始数据集行数: {len(df)}")
        
        df["Sequence"] = df["Sequence"].apply(clean_sequence)
        df = df.dropna(subset=['Sequence'])
        logger.info(f"氨基酸序列清洗后行数: {len(df)}")

        # 将宽格式转换为长格式
        id_vars = ['ID', 'Sequence', 'Length']
        value_vars = [col for col in df.columns if col.startswith('MIC')]
        long_df = df.melt(
            id_vars=id_vars,
            value_vars=value_vars,
            var_name='Raw_Species',
            value_name='MIC_Value'
        )

        # 转换Raw_Species 格式
        try:
            long_df["Raw_Species"] = long_df["Raw_Species"].apply(extract_raw_species)
        except Exception as e:
            logger.warning(f"提取Raw_Species失败: {e}")

        # MIC值清洗
        long_df = long_df.dropna(subset=['MIC_Value'])
        logger.info(f"MIC值存在的行数: {len(long_df)}")
        
        long_df["MIC_Value(ug/ml)"] = long_df.apply(unify_to_ug_ml, axis=1)
        long_df = long_df.dropna(subset=['MIC_Value(ug/ml)'])
        logger.info(f"MIC值清洗后行数: {len(long_df)}")
    else:
        long_df = pd.read_csv(csv2process, dtype={"ID": str})
        logger.info(f"读取处理后的数据集，行数: {len(long_df)}")
    
    # # test
    # long_df = long_df.iloc[:10, :].copy()

    # 提取strain，改写Raw_Species
    if "Strain" not in long_df.columns:
        long_df[["Raw_Species", "Strain"]] = long_df["Raw_Species"].apply(lambda x: pd.Series(split_strain_species(x)))

    # 获取唯一物种列表
    unique_species = long_df["Raw_Species"].drop_duplicates().tolist()
    logger.info(f"唯一物种数: {len(unique_species)}")
    
    # 使用多线程处理物种名称
    name_df_dict = {}
    try:
        with ThreadPoolExecutor(max_workers=arg.workers) as executor:
            futures = {executor.submit(_process_raw_species, sp): sp for sp in unique_species}
            completed = 0
            
            for future in as_completed(futures):
                try:
                    raw, tax_id, target_species = future.result(timeout=30)
                    name_df_dict[raw] = (tax_id, target_species)
                    completed += 1
                    if completed % 10 == 0:
                        logger.info(f"已处理 {completed}/{len(unique_species)} 个物种")
                except Exception as e:
                    sp = futures[future]
                    logger.error(f"处理物种失败 ({sp}): {e}")
                    name_df_dict[sp] = (None, None)
        
        logger.info(f"物种处理完成，共处理 {len(name_df_dict)} 个物种")
    except Exception as e:
        logger.error(f"多线程处理异常: {e}")
        # 回退到顺序处理
        logger.info("回退到顺序处理模式")
        for sp in unique_species:
            tax_id, target_species = extract_bacterium(sp, direct_NCBI_search=True)
            name_df_dict[sp] = (tax_id, target_species)
    
    # 构建名称映射表
    print(name_df_dict)
    name_df = pd.DataFrame.from_dict(
        name_df_dict, 
        orient='index', 
        columns=['Tax_ID', 'Target_Species']
    ).reset_index().rename(columns={'index': 'raw'})
    print(name_df.head(5))
    # 合并数据
    long_df = long_df.merge(name_df, left_on="Raw_Species", right_on="raw", how="left")
    print(long_df.head(5))
    # APD6_MIC_2.csv 里面本来就有Tax_ID和Target_Species列，覆盖掉
    if "Tax_ID_y" in long_df.columns and "Target_Species_y" in long_df.columns:
        long_df.rename(columns={"Tax_ID_y": "Tax_ID", "Target_Species_y": "Target_Species"}, inplace=True)
    long_df = long_df[["ID", "Sequence", "Length", "log_MIC_Value(uM)", "Tax_ID", "Raw_Species", "Target_Species", "Strain"]]

    output_file = os.path.join(arg.output_path, "final_" + arg.input_file)
    long_df.to_csv(output_file, index=False)
    logger.info(f"处理完成，输出文件: {output_file}")



if __name__ == "__main__":
    main()