import re

file_path = "/home/jimchen/miniconda3/envs/iprvllm/lib/python3.9/site-packages/vllm/engine/arg_utils.py"

with open(file_path, 'r') as f:
    content = f.read()

# 1. 移除 deprecated 参数及其赋值 (处理各种空格和换行)
# 匹配类似 , deprecated="xxx" 或 , deprecated=True
content = re.sub(r',\s*deprecated=[^,)]+', '', content)
# 处理括号起始位置的情况 (deprecated=xxx, ...)
content = re.sub(r'\(deprecated=[^,)]+,\s*', '(', content)

# 2. 修复可能产生的语法漏洞
content = content.replace(", ,", ",")
content = content.replace("(,", "(")

with open(file_path, 'w') as f:
    f.write(content)

print("vLLM arg_utils.py fixed successfully.")
