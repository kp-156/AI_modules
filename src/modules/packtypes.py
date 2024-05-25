from docarray import BaseDoc
from typing import Optional, Dict, List

class TranscriptionPack(BaseDoc):
    full_text: Optional[str] = None
    full_dialogue: Optional[str] = None
    speaker1_full_text: Optional[str] = None
    speaker2_full_text: Optional[str] = None
    speaker1_dialogue: Optional[List[str]] = None
    speaker2_dialogue: Optional[List[str]] = None
    overall_sentiment: Optional[str] = None
    has_positive: Optional[bool] = None
    has_negative: Optional[bool] = None
    positive_sentence_list: Optional[List[str]] = None
    negative_sentence_list: Optional[List[str]] = None
    transcription_score: Optional[Dict] = None

class TranslationPack(BaseDoc):
    text: Optional[str]
    translation: Optional[str]
    source: Optional[str]
    target: Optional[str] = 'en'
    status: Optional[str] = 'To Translate!'

class ImagePack(BaseDoc):
    name : Optional[str]
    text : Optional[str]
    bytes_ : Optional[bytes]
    status : Optional[str] = 'To easyocr!'

class Pack(BaseDoc):
    cat : str
    lst : bytes