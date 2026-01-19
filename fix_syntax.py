import re

file_path = "/home/jimchen/miniconda3/envs/iprvllm/lib/python3.9/site-packages/vllm/engine/arg_utils.py"

with open(file_path, 'r') as f:
    lines = f.readlines()

fixed_lines = []
for line in lines:
    # 移除可能残留在行内的 deprecated 参数及其赋值
    line = re.sub(r'deprecated=[^,)]+', '', line)
    fixed_lines.append(line)

content = "".join(fixed_lines)

# 核心：清理因为删除而产生的语法错误
content = re.sub(r',\s*,', ',', content)      # 将 ", ," 变成 ","
content = re.sub(r'\(\s*,', '(', content)     # 将 "( ," 变成 "("
content = re.sub(r',\s*\)', ')', content)     # 将 ", )" 变成 ")"
content = re.sub(r'^\s*,\s*$', '', content, flags=re.MULTILINE) # 移除孤立在某一行的逗号

with open(file_path, 'w') as f:
    f.write(content)

print("vLLM syntax fixed. No more lonely commas.")
