"""
Multi-language support and internationalization capabilities for RAG system.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import json
from pathlib import Path

# Optional imports with fallbacks
try:
    from langdetect import detect, detect_langs
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    detect = None
    detect_langs = None

try:
    from googletrans import Translator
    GOOGLETRANS_AVAILABLE = True
except ImportError:
    GOOGLETRANS_AVAILABLE = False
    Translator = None

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None

logger = logging.getLogger(__name__)

@dataclass
class LanguageDetectionResult:
    """Result of language detection."""
    language: str
    confidence: float
    alternatives: List[Tuple[str, float]]

@dataclass
class TranslationResult:
    """Result of translation."""
    original_text: str
    translated_text: str
    source_language: str
    target_language: str
    confidence: float

class LanguageDetector:
    """Language detection service."""
    
    def __init__(self):
        self.available = LANGDETECT_AVAILABLE
        
        if not self.available:
            logger.warning("Language detection not available. Install langdetect.")
        else:
            logger.info("LanguageDetector initialized")
    
    def detect_language(self, text: str, include_alternatives: bool = False) -> Optional[LanguageDetectionResult]:
        """Detect language of text."""
        if not self.available or not text.strip():
            return None
        
        try:
            if include_alternatives:
                # Get multiple language predictions
                lang_probs = detect_langs(text)
                primary = lang_probs[0]
                alternatives = [(lang.lang, lang.prob) for lang in lang_probs[1:5]]
                
                return LanguageDetectionResult(
                    language=primary.lang,
                    confidence=primary.prob,
                    alternatives=alternatives
                )
            else:
                # Simple detection
                detected_lang = detect(text)
                return LanguageDetectionResult(
                    language=detected_lang,
                    confidence=0.9,  # langdetect doesn't provide confidence for simple detection
                    alternatives=[]
                )
                
        except Exception as e:
            logger.error(f"Language detection failed: {e}")
            return None
    
    def is_supported_language(self, language_code: str) -> bool:
        """Check if language is supported."""
        # Common languages supported by most models
        supported_languages = {
            'en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'zh', 'ja', 'ko',
            'ar', 'hi', 'th', 'vi', 'tr', 'pl', 'nl', 'sv', 'da', 'no'
        }
        return language_code in supported_languages

class TranslationService:
    """Translation service with multiple backends."""
    
    def __init__(self, preferred_service: str = "google"):
        self.preferred_service = preferred_service
        self.google_translator = None
        
        if GOOGLETRANS_AVAILABLE:
            try:
                self.google_translator = Translator()
                logger.info("Google Translate service initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Google Translate: {e}")
        else:
            logger.warning("Google Translate not available. Install googletrans.")
    
    def translate(self, text: str, target_language: str, 
                 source_language: Optional[str] = None) -> Optional[TranslationResult]:
        """Translate text to target language."""
        if not text.strip():
            return None
        
        if self.preferred_service == "google" and self.google_translator:
            return self._translate_google(text, target_language, source_language)
        else:
            logger.warning("No translation service available")
            return None
    
    def _translate_google(self, text: str, target_language: str, 
                         source_language: Optional[str] = None) -> Optional[TranslationResult]:
        """Translate using Google Translate."""
        try:
            result = self.google_translator.translate(
                text, 
                dest=target_language,
                src=source_language
            )
            
            return TranslationResult(
                original_text=text,
                translated_text=result.text,
                source_language=result.src,
                target_language=target_language,
                confidence=0.9  # Google Translate doesn't provide confidence scores
            )
            
        except Exception as e:
            logger.error(f"Google translation failed: {e}")
            return None
    
    def get_supported_languages(self) -> Dict[str, str]:
        """Get supported languages."""
        # Common language codes and names
        return {
            'en': 'English',
            'es': 'Spanish',
            'fr': 'French',
            'de': 'German',
            'it': 'Italian',
            'pt': 'Portuguese',
            'ru': 'Russian',
            'zh': 'Chinese',
            'ja': 'Japanese',
            'ko': 'Korean',
            'ar': 'Arabic',
            'hi': 'Hindi',
            'th': 'Thai',
            'vi': 'Vietnamese',
            'tr': 'Turkish',
            'pl': 'Polish',
            'nl': 'Dutch',
            'sv': 'Swedish',
            'da': 'Danish',
            'no': 'Norwegian'
        }

class MultilingualEmbedder:
    """Multilingual embedding service."""
    
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        self.model_name = model_name
        self.model = None
        self.available = SENTENCE_TRANSFORMERS_AVAILABLE
        
        if not self.available:
            logger.warning("Sentence transformers not available")
            return
        
        try:
            self.model = SentenceTransformer(model_name)
            logger.info(f"Multilingual embedder initialized with {model_name}")
        except Exception as e:
            logger.error(f"Failed to load multilingual model: {e}")
            self.model = None
            self.available = False
    
    def encode(self, texts: List[str]) -> Optional[List[List[float]]]:
        """Encode texts into multilingual embeddings."""
        if not self.available or not self.model:
            return None
        
        try:
            embeddings = self.model.encode(texts)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Encoding failed: {e}")
            return None
    
    def encode_single(self, text: str) -> Optional[List[float]]:
        """Encode single text."""
        result = self.encode([text])
        return result[0] if result else None

class LocalizationManager:
    """Manages localized strings and templates."""
    
    def __init__(self, locales_path: str = "locales"):
        self.locales_path = Path(locales_path)
        self.locales_path.mkdir(exist_ok=True)
        
        self.translations: Dict[str, Dict[str, str]] = {}
        self.default_language = 'en'
        
        # Load translations
        self._load_translations()
        
        logger.info(f"LocalizationManager initialized with {len(self.translations)} languages")
    
    def get_text(self, key: str, language: str = None, **kwargs) -> str:
        """Get localized text."""
        language = language or self.default_language
        
        # Try requested language first
        if language in self.translations and key in self.translations[language]:
            text = self.translations[language][key]
        # Fallback to default language
        elif self.default_language in self.translations and key in self.translations[self.default_language]:
            text = self.translations[self.default_language][key]
        # Return key if not found
        else:
            text = key
        
        # Format with kwargs
        try:
            return text.format(**kwargs)
        except (KeyError, ValueError):
            return text
    
    def add_translation(self, language: str, key: str, text: str):
        """Add a translation."""
        if language not in self.translations:
            self.translations[language] = {}
        
        self.translations[language][key] = text
        self._save_translation_file(language)
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages."""
        return list(self.translations.keys())
    
    def _load_translations(self):
        """Load translation files."""
        for locale_file in self.locales_path.glob("*.json"):
            language = locale_file.stem
            
            try:
                with open(locale_file, 'r', encoding='utf-8') as f:
                    self.translations[language] = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load locale {language}: {e}")
        
        # Create default English translations if none exist
        if 'en' not in self.translations:
            self._create_default_translations()
    
    def _create_default_translations(self):
        """Create default English translations."""
        default_translations = {
            'welcome_message': 'Welcome to the Customer Support Assistant',
            'ask_question': 'Ask a question',
            'generating_response': 'Generating response...',
            'no_results_found': 'No relevant information found',
            'error_occurred': 'An error occurred while processing your request',
            'language_detected': 'Detected language: {language}',
            'translation_available': 'Translation available in your language',
            'feedback_thanks': 'Thank you for your feedback!',
            'response_helpful': 'Was this response helpful?',
            'yes': 'Yes',
            'no': 'No',
            'submit_feedback': 'Submit Feedback',
            'conversation_history': 'Conversation History',
            'citations': 'Citations',
            'response_time': 'Response Time: {time}s',
            'language_support': 'Language Support',
            'auto_translate': 'Auto-translate responses',
            'original_language': 'Original Language',
            'translated_from': 'Translated from {language}'
        }
        
        self.translations['en'] = default_translations
        self._save_translation_file('en')
    
    def _save_translation_file(self, language: str):
        """Save translation file."""
        locale_file = self.locales_path / f"{language}.json"
        
        try:
            with open(locale_file, 'w', encoding='utf-8') as f:
                json.dump(self.translations[language], f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save locale {language}: {e}")

class MultilingualRAGSystem:
    """Multilingual RAG system coordinator."""
    
    def __init__(self, base_retriever, base_llm, 
                 auto_translate: bool = True,
                 target_language: str = 'en'):
        self.base_retriever = base_retriever
        self.base_llm = base_llm
        self.auto_translate = auto_translate
        self.target_language = target_language
        
        # Initialize components
        self.language_detector = LanguageDetector()
        self.translation_service = TranslationService()
        self.multilingual_embedder = MultilingualEmbedder()
        self.localization = LocalizationManager()
        
        logger.info("MultilingualRAGSystem initialized")
    
    def process_query(self, query: str, user_language: Optional[str] = None,
                     top_k: int = 10) -> Dict[str, Any]:
        """Process query with multilingual support."""
        # Detect language if not provided
        detected_language = None
        if not user_language:
            detection_result = self.language_detector.detect_language(query, include_alternatives=True)
            if detection_result:
                user_language = detection_result.language
                detected_language = detection_result
        
        original_query = query
        query_language = user_language or 'en'
        
        # Translate query to English for retrieval if needed
        translated_query = None
        if query_language != self.target_language and self.auto_translate:
            translation_result = self.translation_service.translate(
                query, self.target_language, query_language
            )
            if translation_result:
                query = translation_result.translated_text
                translated_query = translation_result
        
        # Perform retrieval
        try:
            retrieved_docs = self.base_retriever.search(query, top_k)
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            retrieved_docs = []
        
        # Generate response
        response_text = ""
        if retrieved_docs:
            try:
                # Build prompt (this would use your existing prompt building logic)
                from rag.chain import build_prompt, ContextDoc
                contexts = [ContextDoc(**doc) for doc in retrieved_docs]
                prompt = build_prompt(query, contexts)
                
                # Generate response
                response_text = self.base_llm.generate(prompt)
                
            except Exception as e:
                logger.error(f"Response generation failed: {e}")
                response_text = self.localization.get_text('error_occurred', query_language)
        else:
            response_text = self.localization.get_text('no_results_found', query_language)
        
        # Translate response back to user language if needed
        translated_response = None
        if (query_language != self.target_language and 
            self.auto_translate and response_text):
            
            translation_result = self.translation_service.translate(
                response_text, query_language, self.target_language
            )
            if translation_result:
                response_text = translation_result.translated_text
                translated_response = translation_result
        
        # Prepare response
        response = {
            'answer': response_text,
            'original_query': original_query,
            'processed_query': query,
            'user_language': query_language,
            'citations': retrieved_docs,
            'language_info': {
                'detected_language': detected_language.__dict__ if detected_language else None,
                'query_translated': translated_query.__dict__ if translated_query else None,
                'response_translated': translated_response.__dict__ if translated_response else None,
                'auto_translate_enabled': self.auto_translate
            },
            'localized_strings': self._get_localized_ui_strings(query_language)
        }
        
        return response
    
    def _get_localized_ui_strings(self, language: str) -> Dict[str, str]:
        """Get localized UI strings."""
        return {
            'response_helpful': self.localization.get_text('response_helpful', language),
            'yes': self.localization.get_text('yes', language),
            'no': self.localization.get_text('no', language),
            'submit_feedback': self.localization.get_text('submit_feedback', language),
            'citations': self.localization.get_text('citations', language),
            'conversation_history': self.localization.get_text('conversation_history', language)
        }
    
    def get_language_stats(self) -> Dict[str, Any]:
        """Get multilingual system statistics."""
        return {
            'language_detector_available': self.language_detector.available,
            'translation_service_available': bool(self.translation_service.google_translator),
            'multilingual_embedder_available': self.multilingual_embedder.available,
            'supported_languages': self.translation_service.get_supported_languages(),
            'localized_languages': self.localization.get_supported_languages(),
            'auto_translate_enabled': self.auto_translate,
            'target_language': self.target_language
        }
    
    def update_localization(self, language: str, translations: Dict[str, str]):
        """Update localization strings."""
        for key, text in translations.items():
            self.localization.add_translation(language, key, text)
        
        logger.info(f"Updated {len(translations)} translations for {language}")

# Language configuration templates
LANGUAGE_CONFIGS = {
    'minimal': {
        'auto_translate': False,
        'target_language': 'en',
        'supported_languages': ['en']
    },
    'basic_multilingual': {
        'auto_translate': True,
        'target_language': 'en',
        'supported_languages': ['en', 'es', 'fr', 'de']
    },
    'full_multilingual': {
        'auto_translate': True,
        'target_language': 'en',
        'supported_languages': [
            'en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'zh', 'ja', 'ko',
            'ar', 'hi', 'th', 'vi', 'tr', 'pl', 'nl'
        ]
    }
}

def create_multilingual_system(base_retriever, base_llm, config_name: str = 'basic_multilingual'):
    """Factory function to create multilingual system."""
    config = LANGUAGE_CONFIGS.get(config_name, LANGUAGE_CONFIGS['basic_multilingual'])
    
    return MultilingualRAGSystem(
        base_retriever=base_retriever,
        base_llm=base_llm,
        auto_translate=config['auto_translate'],
        target_language=config['target_language']
    )
