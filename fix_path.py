import sys
import glob

def fix_file(filename):
    with open(filename, "r") as f:
        content = f.read()
    
    if "import sys" in content and "sys.path.insert" in content:
        return
        
    if "mardpg_uav" not in content and "scripts." not in content:
        return
        
    lines = content.split('\n')
    
    # Check if there is an import block
    import_idx = 0
    has_os = False
    for i, line in enumerate(lines):
        if line.startswith("import os"):
            import_idx = i
            has_os = True
            break
        elif line.startswith("import") or line.startswith("from"):
            import_idx = i - 1 if i > 0 else 0
            break
            
    insert_lines = []
    if not has_os:
        insert_lines.append("import os")
    if "import sys" not in content:
        insert_lines.append("import sys")
    insert_lines.append("sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))")
    
    for i, line in enumerate(insert_lines):
        lines.insert(import_idx + i, line)
        
    with open(filename, "w") as f:
        f.write('\n'.join(lines))
    print(f"Fixed {filename}")

for f in glob.glob("mardpg-uav/scripts/*.py"):
    fix_file(f)

