#!/usr/bin/env python3
"""
Validation script to verify MySQL connection timeout fixes are present in the code.
This script checks that the necessary changes were made without importing the module.
"""

import re
from pathlib import Path

def check_mysql_fixes():
    """Verify that MySQL connection timeout fixes are present in the code."""
    
    script_path = Path(__file__).resolve().parent.parent / 'resume_matcher_rag.py'
    
    with open(script_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    checks = {
        'read_timeout': False,
        'write_timeout': False,
        'autocommit': False,
        'use_pure': False,
        'retry_logic': False,
        'exponential_backoff': False,
    }
    
    # Check for read_timeout parameter
    if re.search(r'read_timeout["\s]*[:=]\s*\d+', content):
        checks['read_timeout'] = True
        print("✓ read_timeout parameter found")
    else:
        print("✗ read_timeout parameter NOT found")
    
    # Check for write_timeout parameter
    if re.search(r'write_timeout["\s]*[:=]\s*\d+', content):
        checks['write_timeout'] = True
        print("✓ write_timeout parameter found")
    else:
        print("✗ write_timeout parameter NOT found")
    
    # Check for autocommit parameter
    if re.search(r'autocommit["\s]*[:=]\s*True', content):
        checks['autocommit'] = True
        print("✓ autocommit parameter found")
    else:
        print("✗ autocommit parameter NOT found")
    
    # Check for use_pure parameter
    if re.search(r'use_pure["\s]*[:=]\s*True', content):
        checks['use_pure'] = True
        print("✓ use_pure parameter found")
    else:
        print("✗ use_pure parameter NOT found")
    
    # Check for retry logic
    if re.search(r'max_retries\s*=\s*\d+', content):
        checks['retry_logic'] = True
        print("✓ Retry logic found")
    else:
        print("✗ Retry logic NOT found")
    
    # Check for exponential backoff
    if re.search(r'retry_delay\s*\*\s*\(', content) or re.search(r'exponential', content, re.IGNORECASE):
        checks['exponential_backoff'] = True
        print("✓ Exponential backoff found")
    else:
        print("✗ Exponential backoff NOT found")
    
    # Count occurrences of read_timeout
    read_timeout_count = len(re.findall(r'read_timeout["\s]*[:=]', content))
    print(f"\nTotal read_timeout occurrences: {read_timeout_count}")
    
    # Check specific functions were updated
    functions_to_check = [
        '_sync_from_mysql',
        '_test_mysql_connection',
        '_peek_mysql_rows',
        'chroma_upsert_from_mysql',
        'mysql_fetch_blob_by_id'
    ]
    
    print("\nFunction-specific checks:")
    for func_name in functions_to_check:
        # Find function definition
        func_match = re.search(rf'def {func_name}\([^)]*\):', content)
        if func_match:
            # Get content after function definition (next 1000 chars should include connection setup)
            func_start = func_match.start()
            func_content = content[func_start:func_start + 3000]
            
            has_read_timeout = 'read_timeout' in func_content
            has_autocommit = 'autocommit' in func_content
            
            if has_read_timeout and has_autocommit:
                print(f"  ✓ {func_name}: Updated with timeout parameters")
            else:
                print(f"  ⚠ {func_name}: May not have all parameters (read_timeout: {has_read_timeout}, autocommit: {has_autocommit})")
    
    # Overall summary
    all_passed = all(checks.values())
    print("\n" + "="*60)
    if all_passed:
        print("✓ ALL CHECKS PASSED - MySQL connection fixes are properly implemented")
    else:
        print("✗ SOME CHECKS FAILED - Review the implementation")
        failed_checks = [k for k, v in checks.items() if not v]
        print(f"  Failed: {', '.join(failed_checks)}")
    print("="*60)
    
    return all_passed


if __name__ == '__main__':
    success = check_mysql_fixes()
    exit(0 if success else 1)
