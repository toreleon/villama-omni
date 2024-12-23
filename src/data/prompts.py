INSTRUCTION_REWRITING="""Translate the input instruction into Vietnamese before generating a speech-like version of it. The speech-like version should simulate human speech with fillers and ensure compatibility for TTS synthesis.

Below is the instruction data containing the user's original instruction. Follow these requirements:

1. **Translate** the instruction into Vietnamese.
2. **Modify** the translated instruction to simulate human speech, adding fillers as appropriate (but not too many like 'you know', 'like', etc.).
3. Ensure content is suitable for TTS synthesis. Numbers should be written in English words instead of Arabic numerals.
4. Keep the instruction relatively brief without excessive verbiage.
    
[instruction]: {instruction}

# Output Format

Please output in JSON format as follows: {{"question" translated_speech_version}}.

# Notes

- Ensure the translation accurately conveys the original meaning.
- Focus on keeping the translated speech natural and easily synthesizable by TTS systems."""