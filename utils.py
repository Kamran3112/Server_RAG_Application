def beautify_response(text: str) -> str:
    lines = text.split('\n')
    cleaned = [line.strip() for line in lines if line.strip()]
    return '\n\n'.join(cleaned)  # double \n for readable paragraph separation
