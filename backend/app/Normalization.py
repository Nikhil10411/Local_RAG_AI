import re

def get_logical_group_id(filename: str) -> str:
    """
    Normalizes filenames by removing version markers like (1), _v2, copy, 
    and file extensions, mapping variations to a single logical identifier.
    """
    # Extract base name without extension
    base_name = filename.rsplit(".", 1)[0].lower()
    
    # Strip common version suffixes like ' (1)', '_v2', '-copy', 'updated'
    cleaned = re.sub(r'\s*[\(\_\-]?\b(v\d+|\d+|copy|updated|revised)\b[\)\_\-]?', '', base_name, flags=re.IGNORECASE)
    
    # Normalize spaces and special characters to underscores
    logical_id = re.sub(r'[\s\(\)\-\.]+', '_', cleaned).strip('_')
    
    return logical_id or base_name