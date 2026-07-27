import json
import pandas as pd
import re
from Bio.SeqUtils import molecular_weight
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_numeric_value(activity_str:str) -> float:
    """
    从Activity字符串中提取数值
    例如: "0.79-12.67" -> 取平均值或首个值
    """
    if '-' in activity_str or "->" in activity_str:
        bounds = activity_str.split('-')

        def _geometric_mean(b0, b1):
            return (b0 * b1) ** (0.5)

        try:
            b0, b1 = float(bounds[0]), float(bounds[1].strip(">"))
            return _geometric_mean(b0, b1)  
        except:
            print(activity_str)
            return bounds[0]
    elif "±" in activity_str:
        return float(activity_str.split("±")[0])
    else:
        # 用正则表达式提取数字
        match = re.search(r'[\d\.]+', str(activity_str))
        if match:
            return float(match.group())
        return None
   

def convert_uM_to_ug_ml(value_uM:float, sequence:str)->float:
    """
    将uM单位转换为ug/ml
    公式: (uM * MW) / 1000 = ug/ml
    其中MW为蛋白质分子量（Da）
    """
    try:
        if value_uM is None:
            return None
        
        # 计算分子量
        mw = molecular_weight(sequence, seq_type="protein")
        # 转换
        return (value_uM * mw) / 1000
    except Exception as e:
        logger.warning(f"分子量计算失败 ({sequence}): {e}")
        return None

def parse_dbaasp_json(json_file):
    """
    解析DBAASP JSON文件并转换为DataFrame
    
    每一行对应一条有MIC值的记录
    列: ID, Sequence, Length, Raw_Species, MIC_Value(ug/ml)
    """
    records = []
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    peptide_count = 0
    mic_count = 0
    
    # 遍历每个肽链
    for sequence_key, peptide_info in data.items():
        peptide_count += 1
        
        # 获取化学结构信息
        chemical_structures = peptide_info.get('chemical_structure', [])
        activity_data = peptide_info.get('activity_data', [])
        
        if not chemical_structures or not activity_data:
            continue
        
        # 取第一个化学结构（通常会有重复）
        chem_struct = chemical_structures[0]
        peptide_id = chem_struct.get('ID')
        sequence = chem_struct.get('Sequence')
        # 清洗序列，保留20种标准氨基酸
        seq = sequence.upper().strip()
        standard_aa = set("ACDEFGHIKLMNPQRSTVWY")
        if not set(seq).issubset(standard_aa):
            logger.warning(f"序列包含非标准氨基酸: {sequence}")
            continue

        length = chem_struct.get('Length')
        
        # 过滤MIC数据
        for activity in activity_data:
            # 只处理MIC数据
            if activity.get('Activity Measure') != 'MIC':
                continue
            
            target_species = activity.get('Target Species')
            activity_value = activity.get('Activity')
            unit = activity.get('Unit', 'µM')
            
            # 提取数值
            mic_value = extract_numeric_value(activity_value)
            if mic_value is None:
                continue
            
            # 单位转换
            # µM 或 uM 都转换为 ug/ml
            if unit.lower() in ['µm', 'um','µM','uM']:
                mic_value_ug_ml = convert_uM_to_ug_ml(mic_value, sequence)
            elif unit.lower() in ['µg/ml', 'ug/ml', 'μg/ml']:
                mic_value_ug_ml = mic_value
            else:
                logger.warning(f"未知单位: {unit}")
                continue
            
            if mic_value_ug_ml is not None:
                records.append({
                    'ID': peptide_id,
                    'Sequence': sequence,
                    'Length': length,
                    'Raw_Species': target_species,
                    'MIC_Value(ug/ml)': mic_value_ug_ml
                })
                mic_count += 1
    
    logger.info(f"总肽链数: {peptide_count}")
    logger.info(f"提取的MIC记录数: {mic_count}")
    
    return pd.DataFrame(records)

if __name__ == "__main__":
    # 转换JSON为DataFrame
    json_file = "DBAASP.data.json"
    output_file = "DBAASP_MIC.csv"
    
    logger.info(f"开始解析 {json_file}...")
    df = parse_dbaasp_json(json_file)
    
    logger.info(f"DataFrame shape: {df.shape}")
    print(df.head(10))
    
    # 保存为CSV
    df.to_csv(output_file, index=False)
    logger.info(f"保存为 {output_file}")
