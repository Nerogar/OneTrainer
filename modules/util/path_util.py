def safe_filename(text: str):
    legal_chars = [' ', '.', '_', '-']
    return ''.join(filter(lambda x: str.isalnum(x) or x in legal_chars, text))[0:32]
