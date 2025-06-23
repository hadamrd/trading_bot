import re
from pathlib import Path

def migrate_prints_to_logs(file_path: Path):
    """Quick script to migrate print statements to logging"""
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Add logger import if not present
    if 'import logging' not in content and 'print(' in content:
        content = 'import logging\nlogger = logging.getLogger(__name__)\n\n' + content
    
    # Replace common patterns
    patterns = [
        (r'print\(f?"✅([^"]+)"\)', r'logger.info("\1")'),
        (r'print\(f?"❌([^"]+)"\)', r'logger.error("\1")'),
        (r'print\(f?"⚠️([^"]+)"\)', r'logger.warning("\1")'),
        (r'print\(f?"🔍([^"]+)"\)', r'logger.debug("\1")'),
        (r'print\(f?"([^"]+)"\)', r'logger.info("\1")'),
    ]
    
    for pattern, replacement in patterns:
        content = re.sub(pattern, replacement, content)
    
    # Save back
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f"Migrated {file_path}")
