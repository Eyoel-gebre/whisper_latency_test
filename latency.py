import whisper
import time

tiny_model = whisper.load_model("tiny")
# base_model = whisper.load_model("base")
# medium_model = whisper.load_model("medium")


start = time.time()
result = tiny_model.transcribe("long_speech.mp3")
print(f'\n\ntiny model finished in {time.time() - start} ms\n')

# start = time.time()
# result = base_model.transcribe("long_speech.mp3")
# print(f'base model finished in {time.time() - start} ms\n')

# start = time.time()
# result = medium_model.transcribe("long_speech.mp3")
# print(f'medium model finished in {time.time() - start} ms\n')


# print statistics such as number of words, number of characters, number of sentences, length of the audio, etc.
print(f'number of words: {len(result["text"])}\n')

# print the text
print('Transcribed text:')
print(result["text"])