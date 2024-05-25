import googletrans

# Translator class for detecting language and translating text
class Translator:
    def __init__(self):
        self.translator = googletrans.Translator()

    def detect_language(self, user_input):
        # Detects the language of the given user input
        try:
            detected_language = self.translator.detect(user_input).lang
            return detected_language
        except Exception as e:
            print(f"Error detecting language: {e}")
            return None

    def translate_user_input(self, user_input, base_language="en"):
        detected_language = self.detect_language(user_input)
        
        if detected_language == base_language:
            return user_input
        else:
            try:
                translated_input = self.translator.translate(user_input, dest=base_language).text
                return translated_input
            except Exception as e:
                print(f"Translation error: {e}")
                return user_input  # Fallback to the original input

    def translate_response(self, response, target_language):
        # Translates the response to the desired language
        try:
            translated_response = self.translator.translate(response, dest=target_language).text
            return translated_response
        except Exception as e:
            print(f"Translation error: {e}")
            return response  # Return original text if translation fails
