from langdetect import detect, LangDetectException
from langdetect.detector_factory import DetectorFactory

DetectorFactory.seed = 0

def is_english(text: str) -> bool:
    try:
        return detect(text) == "en"
    except LangDetectException:
        return False