import os
import re
from pathlib import Path

def test_no_hardcoded_user_paths():
    """
    Scans all non-test Python files for forbidden hardcoded paths like /Users/, .gemini, Downloads.
    """
    forbidden_patterns = [
        re.compile(r'/Users/\w+'),
        re.compile(r'\.gemini'),
        re.compile(r'/Downloads/')
    ]
    
    # repo root from tests/
    repo_root = Path(__file__).resolve().parent.parent
    
    violations = []
    
    for py_file in repo_root.rglob('*.py'):
        # skip tests directory and venv
        if 'tests/' in py_file.as_posix() or 'venv/' in py_file.as_posix() or '.venv/' in py_file.as_posix():
            continue
            
        with open(py_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        for idx, line in enumerate(lines):
            # Ignore comments just to be safe, although we shouldn't have them there either
            if line.strip().startswith('#'):
                continue
                
            for pattern in forbidden_patterns:
                if pattern.search(line):
                    violations.append(f"{py_file.relative_to(repo_root)}:{idx+1} -> {line.strip()}")
                    
    assert len(violations) == 0, "Found hardcoded paths in source files:\n" + "\n".join(violations)
