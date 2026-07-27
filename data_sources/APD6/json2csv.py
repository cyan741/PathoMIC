import json
import re
import pandas as pd
from typing import Dict
import html 
bacteria_regex = r'(?P<bacterium>[A-Z]\. [a-z]+)(?P<strain>\s?[A-Z]+\s?[0-9]+)?'

def get_data(data_string:str) -> Dict:

    all_bacteria = {}
    unit_strings = ['uM', 'ug/ml']
    unit_delimiters = [u + ')' for u in unit_strings]
    unit_regex_pattern = '(' + '|'.join(map(re.escape, unit_delimiters)) + ')'
    fields = re.split(unit_regex_pattern, data_string)
    fields_and_units = []
    for i, f in enumerate(fields[:-1]):  # Parentheses around regex mean match expression is also returned
        # Up to -1 to skip the last one, which has no corresponding unit match
        if i % 2 == 0:
            fields_and_units.append({'bacteria_string': f, 'unit': fields[i+1]})

    for field_and_unit in fields_and_units:

        bacteria_string = field_and_unit['bacteria_string']
        unit = field_and_unit['unit']
        
        bacteria_matches = re.finditer(bacteria_regex, bacteria_string)
        
        def _extract_value(range_expr):
            if '-' not in range_expr:
                return range_expr
            bounds = range_expr.split('-')

            def _geometric_mean(b0, b1):
                return (b0 * b1) ** (0.5)

            try:
                b0, b1 = float(bounds[0]), float(bounds[1])
                return str(_geometric_mean(b0, b1))  
            except:
                print("?????")
                return bounds[0]
                
        
        numeric_range_regex = r'MIC \d+\.?\s?\-?\s?\d*'
        numeric_match = re.search(numeric_range_regex, bacteria_string)
        if not numeric_match:
            continue
        mic_match_string = numeric_match.group(0)
        numeric_part = mic_match_string[4:]
        value = _extract_value(numeric_part)
        for bacteria_match in bacteria_matches:
            bacterium = bacteria_match.groupdict()['bacterium']
            strain = bacteria_match.groupdict()['strain']
            if strain:
                strain = strain.strip()
            all_bacteria[(bacterium, strain)] = {
                'unit': unit,
                'value': value,
            }
            
    return all_bacteria

def extract_hc50(additional_info:str) -> Dict:
    """
    从Additional_info中提取HC50信息，返回字典
    结构类似于get_data函数返回的格式
    """
    all_hc50 = {}
    
    # 匹配HC50信息，格式如：HC50 >128 ug/ml 或 HC50 128 ug/ml
    hc50_pattern = r'(?P<cell_type>[\w\s]+):\s*(?P<metric>HC50)\s*(?P<operator>[><=]+)\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>[\w/]+)'
    decoded_sting = html.unescape(additional_info)
    matches = re.finditer(hc50_pattern, decoded_sting)
    
    for match in matches:
        cell_type = match.groupdict()["cell_type"]
        operator = match.groupdict()["operator"]
        value = match.groupdict()["value"]
        unit = match.groupdict()["unit"]
        
        # 如果有大于/小于符号，保存为文本
        if operator:
            value_str = f"{operator}{value}"
        else:
            value_str = value
            
        all_hc50[cell_type] = {
            'unit': unit,
            'value': value_str
        }
    
    return all_hc50

def process_apd6_json(json_file:str, output_csv:str) -> pd.DataFrame:
    """
    处理APD6_data.json文件，提取并整理数据
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 收集所有唯一的菌株和毒性数据列名
    all_bacteria = set()
    all_hc50 = set()
    
    rows = []
    
    for apd_id, entry in data.items():
        if not isinstance(entry, dict):
            continue
            
        # 提取基本信息
        sequence = entry.get('Sequence', '')
        length = entry.get('Length', '')
        activity = entry.get('Activity', '')
        additional_info = entry.get('Additional_info', '')
        
        # 处理MIC数据
        mic_data = get_data(additional_info)
        
        # 处理HC50数据
        hc50_data = extract_hc50(additional_info)
        
        # 添加所有菌株和毒性数据的列名
        for (bacterium, strain), _ in mic_data.items():
            if strain:
                col_name = f"MIC ({bacterium} {strain})"
            else:
                col_name = f"MIC ({bacterium})"
            all_bacteria.add(col_name)
        
        for cell_type, _ in hc50_data.items():
            col_name = f"HC50 ({cell_type})"
            all_hc50.add(col_name)
        
        rows.append({
            'APD_ID': apd_id,
            'Sequence': sequence,
            'Length': length,
            'Activity': activity,
            'Additional_info': additional_info,
            'mic_data': mic_data,
            'hc50_data': hc50_data
        })
    
    df_data = []
    
    for row in rows:
        df_row = {
            'APD_ID': row['APD_ID'],
            'Sequence': row['Sequence'],
            'Length': row['Length'],
            'Activity': row['Activity']
        }
        
        for (bacterium, strain), mic_info in row['mic_data'].items():
            if strain:
                col_name = f"MIC ({bacterium} {strain})"
            else:
                col_name = f"MIC ({bacterium})"
            df_row[col_name] = f"{mic_info['value']} {mic_info['unit']}"
        
        for cell_type, hc50_info in row['hc50_data'].items():
            col_name = f"HC50 ({cell_type})"
            df_row[col_name] = f"{hc50_info['value']} {hc50_info['unit']}"
        
        df_data.append(df_row)
    
    df = pd.DataFrame(df_data)
    
    # 调整列的顺序：基本信息在前，MIC和HC50在后
    basic_cols = ['APD_ID', 'Sequence', 'Length', 'Activity']
    mic_cols = sorted([col for col in df.columns if col.startswith('MIC')])
    hc50_cols = sorted([col for col in df.columns if col.startswith('HC50')])
    
    column_order = basic_cols + mic_cols + hc50_cols
    df = df[column_order]
    
    # 保存到CSV
    df.to_csv(output_csv, index=False, encoding='utf-8')
    
    print(f"数据处理完成！")
    print(f"总共处理了 {len(df)} 条肽数据")
    print(f"提取了 {len(mic_cols)} 种菌株的MIC数据")
    print(f"提取了 {len(hc50_cols)} 种细胞/组织的HC50数据")
    print(f"输出文件：{output_csv}")
    
    return df

if __name__ == '__main__':
    # 设置路径
    json_file = 'APD6_data.json'
    output_csv = 'APD6_processed.csv'
    
    print(f"读取JSON文件：{json_file}")
    df = process_apd6_json(str(json_file), str(output_csv))
    print(f"\n前5行数据预览：")
    print(df.iloc[:5, :4])  # 只显示前4列作为预览
