"""
OneTrainer UI 汉化脚本
读取 zh_cn_map.py 中的翻译映射，替换 Base*.py 和 PySide6*.py 中的英文字符串。
"""
import os
import re
import sys

# 确保 UTF-8
sys.stdout.reconfigure(encoding='utf-8')

UI_DIR = os.path.join(os.path.dirname(__file__), 'modules', 'ui')

def load_translations():
    """加载翻译映射"""
    map_file = os.path.join(os.path.dirname(__file__), 'zh_cn_map.py')
    if not os.path.exists(map_file):
        print(f"Error: {map_file} not found")
        sys.exit(1)
    
    # 动态导入
    import importlib.util
    spec = importlib.util.spec_from_file_location("zh_cn_map", map_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.TRANSLATIONS

def apply_translations(translations, dry_run=False):
    """应用翻译到 UI 文件"""
    stats = {"files_modified": 0, "strings_replaced": 0, "strings_not_found": []}
    
    for filename in sorted(os.listdir(UI_DIR)):
        if not filename.endswith('.py'):
            continue
        # 只处理 Base 和 PySide6 文件
        if not (filename.startswith('Base') or filename.startswith('PySide6')):
            continue
        
        filepath = os.path.join(UI_DIR, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original = content
        file_replacements = 0
        
        for en, zh in translations.items():
            # 匹配带引号的英文字符串
            # 注意：要精确匹配，避免替换到变量名
            pattern = '"' + re.escape(en) + '"'
            replacement = '"' + zh + '"'
            
            new_content = re.sub(pattern, replacement, content)
            if new_content != content:
                count = len(re.findall(pattern, content))
                file_replacements += count
                content = new_content
        
        if content != original:
            if not dry_run:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
            stats["files_modified"] += 1
            stats["strings_replaced"] += file_replacements
            print(f"  {'[DRY] ' if dry_run else ''}{filename}: {file_replacements} replacements")
    
    # 检查未匹配的翻译
    for en in translations:
        found = False
        for filename in os.listdir(UI_DIR):
            if not filename.endswith('.py'):
                continue
            if not (filename.startswith('Base') or filename.startswith('PySide6')):
                continue
            filepath = os.path.join(UI_DIR, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                if f'"{en}"' in f.read():
                    found = True
                    break
        if not found:
            stats["strings_not_found"].append(en)
    
    return stats

def main():
    dry_run = '--dry-run' in sys.argv
    
    print("Loading translations...")
    translations = load_translations()
    print(f"Loaded {len(translations)} translations")
    
    print(f"\n{'[DRY RUN] ' if dry_run else ''}Applying translations to UI files...")
    stats = apply_translations(translations, dry_run)
    
    print(f"\n=== Results ===")
    print(f"Files modified: {stats['files_modified']}")
    print(f"Strings replaced: {stats['strings_replaced']}")
    
    if stats["strings_not_found"]:
        print(f"Strings not found in UI files ({len(stats['strings_not_found'])}):")
        for s in stats["strings_not_found"][:10]:
            print(f"  - {s}")
        if len(stats["strings_not_found"]) > 10:
            print(f"  ... and {len(stats['strings_not_found']) - 10} more")

if __name__ == '__main__':
    main()
