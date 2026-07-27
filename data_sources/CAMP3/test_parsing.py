import re
from math import sqrt

def process_value(value_str):
    """
    处理MIC/IC50值
    - 如果是范围 (如 10-20)，取几何平均
    - 保留单位
    - 保留 > < 等符号
    返回格式: "值 单位" 或 "> 值 单位"
    """
    value_str = value_str.strip()
    
    # 保留比较符号
    comparison_op = ''
    if value_str.startswith('>'):
        comparison_op = '>'
        value_str = value_str[1:].strip()
    elif value_str.startswith('<'):
        comparison_op = '<'
        value_str = value_str[1:].strip()
    elif value_str.startswith('>='):
        comparison_op = '>='
        value_str = value_str[2:].strip()
    elif value_str.startswith('<='):
        comparison_op = '<='
        value_str = value_str[2:].strip()
    
    # 提取单位
    unit_match = re.search(r'(microg/ml|microM|μg/ml|mg/ml|mM|nM|pM)', value_str, re.IGNORECASE)
    unit = unit_match.group(1) if unit_match else ''
    
    # 移除单位，只保留数值部分
    value_only = re.sub(r'(microg/ml|microM|μg/ml|mg/ml|mM|nM|pM)', '', value_str, flags=re.IGNORECASE).strip()
    
    # 移除其他比较符号
    value_only = re.sub(r'[<>=]+', '', value_only).strip()
    
    # 检查是否是范围
    if '-' in value_only:
        # 提取范围的两个数值
        range_match = re.match(r'([\d.]+)\s*-\s*([\d.]+)', value_only)
        if range_match:
            val1 = float(range_match.group(1))
            val2 = float(range_match.group(2))
            # 计算几何平均数
            geometric_mean = sqrt(val1 * val2)
            result = f"{geometric_mean:.2f} {unit}".strip()
        else:
            return None
    else:
        # 单个数值
        number_match = re.search(r'[\d.]+', value_only)
        if number_match:
            result = f"{number_match.group()} {unit}".strip()
        else:
            return None
    
    # 添加比较符号
    if comparison_op:
        result = f"{comparison_op} {result}"
    
    return result

def parse_species_values(target_str, indicator='MIC'):
    """
    从Target字符串中提取物种及其对应的MIC/IC50值
    返回字典: {物种: 值_带单位}
    
    处理多种格式：
    - "物种 ( MIC = value )"
    - "描述 物种 ( MIC = value )"
    - "X % 物种 ( MIC = value )"
    """
    if not target_str or target_str == '':
        return {}
    
    result = {}
    
    # 先按逗号分割，然后在每一段中查找指标
    # 这样可以正确处理物种前有描述的情况
    segments = target_str.split(',')
    
    for segment in segments:
        # 在每一段中查找 MIC/IC50
        pattern = rf'(.+?)\s*\(\s*{indicator}\s*([=><\s]+)\s*([^)]+)\)'
        match = re.search(pattern, segment, re.IGNORECASE)
        
        if match:
            species = match.group(1).strip()
            comparison_op = match.group(2).strip()
            value_str = comparison_op + match.group(3).strip() # 中间没有空格
            
            # 清理物种名称：移除前面的描述性文本
            # 如果包含 %，取 % 之后的部分
            if '%' in species:
                species = species.split('%')[-1].strip()
            
            # 如果还有其他描述性词汇（如数字+空格开头），取最后一部分
            # 这通常是实际的物种名称
            parts = species.split()
            if len(parts) > 2:
                # initial_pattern: C. albicans， S.aureus CCT 6538， E. coli K12， S.?aureus MRSA 
                # fullname_pattern: Staphylococcus aureus NCTC 10571, Mycosphaerella arachidicola，
                for i in range(len(parts) - 1, -1, -1):
                    initial_pattern = (i < len(parts) - 1 and len(parts[i]) > 1 and parts[i][0].isupper() and parts[i][1] == "." )
                    fullname_pattern = (i < len(parts) - 1 and len(parts[i]) > 1 and parts[i][0].isupper() and parts[i][1].islower() and parts[i+1][0].islower() )
                    if initial_pattern  or fullname_pattern:
                        species = ' '.join(parts[i:])
                        break
            
            # 规范化空格
            species = re.sub(r'\s+', ' ', species).strip()
            
            if species and value_str and len(species) > 1:
                # 处理值
                value_with_unit = process_value(value_str)
                if value_with_unit:
                    result[species] = value_with_unit
    
    return result

# 测试用例
test_cases = [
    "S.?aureus ACCT 13883( MIC = >25 microg/ml), S.?aureus MRSA ( MIC > 50 microg/ml ) , E. coli ( MIC = 25 microg/ml ) , P. aeruginosa ( MIC > 50 microg/ml )",
    "20 % growth inhibition F. oxysporum ACCT 13883( MIC = 15.6-40 microM ), 45 % growth inhibition M. arachidicola ( MIC = 15.6 microM ), 63 % growth inhibition P. piricola ( MIC = 15.6 microM )",
]

for test_target in test_cases:
    print("测试字符串:")
    print(test_target)
    print("\n提取结果:")
    results = parse_species_values(test_target, 'MIC')
    for species, value in results.items():
        print(f"  {species}: {value}")
    print()
