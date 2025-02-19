import os

import midi2audio
import librosa
import soundfile as sf

# Get file at https://member.keymusician.com/Member/FluidR3_GM/index.html
SOUNDFONT_PATH = "/scratch/shared/soundfonts/fluid_r3/FluidR3_GM.sf2"
fs = midi2audio.FluidSynth(SOUNDFONT_PATH)

def synthesize_midi(midi_path, save_mp3_only=True):
    wav_path = midi_path.replace(".mid", ".wav")

    fs.midi_to_audio(midi_path, wav_path)

    # trim silence
    wav, sr = librosa.load(wav_path)
    _orig_len = len(wav)
    wav, _ = librosa.effects.trim(wav, top_db=30)
    _new_len = len(wav)
    print(f"Trimmed {(_orig_len - _new_len) / sr:.2f} seconds of silence")
    # os.rename(wav_path, wav_path.replace(".wav", "_orig.wav"))
    sf.write(wav_path, wav, sr)

    if save_mp3_only:
        # use ffmpeg to convert wav to mp3
        os.system(f"ffmpeg -i {wav_path} -codec:a libmp3lame -qscale:a 2 {midi_path.replace('.mid', '.mp3')} -y")
        os.remove(wav_path)