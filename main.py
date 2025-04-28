from mcp.server.fastmcp import FastMCP
import fasttext
import gcld3
from langdetect import detect as langdetect_detect
from langdetect import DetectorFactory
import numpy as np

# Create FastMCP instance
mcp = FastMCP('polyglot-detect')

# Set seed for consistent results with langdetect
DetectorFactory.seed = 0

_model = fasttext.load_model("lid.176.ftz")
_LANG_IDENTIFIER = gcld3.NNetLanguageIdentifier(min_num_bytes=0, max_num_bytes=1000)

def detect_language_langdetect(text: str) -> tuple[str, float]:
    """
    Detect language using langdetect library.
    
    args:
        text (str): Text to detect language of.

    return: Tuple of (language code, confidence)
    """
    try:
        lang = langdetect_detect(text)
        return lang, 0.5  # Langdetect doesn't provide confidence scores
    except Exception as e:
        return "und", 0.0

def detect_language_fasttext(text: str) -> tuple[str, float, list]:
    """
    Detect language of the input and return ISO-639-1 code of the language and confidence.
    
    args:
        text (str): Text to detect language of.

    return: Tuple of (language code, confidence, alternatives list)
    """
    MIN_CONFIDENCE = 0.1
    try:
        labels, probs = _model.predict(text, k=3)
        alternatives = [
            (label.replace('__label__', ''), prob)
            for label, prob in zip(labels, probs)
        ]
        
        if probs[0] >= MIN_CONFIDENCE:
            return labels[0].replace("__label__", ""), probs[0], alternatives
        return "und", 0.0, alternatives
    except Exception as e:
        return "und", 0.0, []

def detect_language_gcld3(text: str) -> tuple[str, float, bool]:
    """
    Detect language of the input and return BCP-47 code of the language and confidence.
    
    args:
        text (str): Text to detect language of.

    return: Tuple of (language code, confidence, is_reliable)
    """
    result = _LANG_IDENTIFIER.FindLanguage(text)
    
    if result is None:
        return "und", 0.0, False
        
    return result.language, result.probability, result.is_reliable

@mcp.tool()
def detect_language(text: str) -> str:
    """
    Detect language using FastText, GCLD3, and langdetect.
    Returns results from all detectors as a markdown formatted string.
    
    args:
        text (str): Text to detect language of.

    return: Markdown formatted string containing results from all detectors
    """
    # Get FastText results with alternatives
    ft_lang, ft_conf, ft_alternatives = detect_language_fasttext(text)
    
    # Get GCLD3 results
    gcld_lang, gcld_conf, is_reliable = detect_language_gcld3(text)
    
    # Get Langdetect results
    ld_lang, ld_conf = detect_language_langdetect(text)
    
    # Format results as markdown
    markdown_result = f"""# Language Detection Results

## FastText
- **Main prediction**: {ft_lang} (confidence: {ft_conf:.3f})
- **Alternative predictions**:
{chr(10).join([f"  - {lang}: {conf:.3f}" for lang, conf in ft_alternatives])}

## GCLD3
- **Language**: {gcld_lang}
- **Confidence**: {gcld_conf:.3f}
- **Is reliable**: {is_reliable}

## Langdetect
- **Language**: {ld_lang}
- **Confidence**: {ld_conf:.3f}
"""
    return markdown_result

if __name__ == "__main__":
    mcp.run(transport="stdio")

