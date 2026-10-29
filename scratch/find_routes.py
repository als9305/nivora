import os

for root, dirs, files in os.walk('.'):
    if any(x in root for x in ['.git', 'node_modules', '__pycache__', '.venv']):
        continue
    for f in files:
        if f.endswith('.py'):
            p = os.path.join(root, f)
            try:
                with open(p, 'r', encoding='utf-8', errors='ignore') as fp:
                    content = fp.read()
                    if 'authenticate_multimodal' in content:
                        print("FOUND authenticate_multimodal in:", p)
            except Exception:
                pass
