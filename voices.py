import subprocess
import os
import shutil
from moviepy import VideoFileClip, AudioFileClip
from pyannote.audio import Pipeline
import warnings
from google import genai

warnings.filterwarnings('ignore')


def separate_vocals_and_other(input_file, output_dir="separated_audio"):
    try:
        subprocess.run(["demucs", "--help"], check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        raise EnvironmentError(
            "Demucs is not installed. Install it with 'pip install demucs'.")

    try:
        subprocess.run(["demucs", "-o", output_dir, input_file], check=True)
        print("Audio separation complete.")
    except subprocess.CalledProcessError as e:
        print("Error during audio separation:", e)
        return None

    separated_folder = os.path.join(
        output_dir, "htdemucs", os.path.splitext(
            os.path.basename(input_file))[0]
    )

    if not os.path.exists(separated_folder):
        print(f"Separation folder not found: {separated_folder}")
        return None

    vocal_file = "vocals.wav"
    src_vocal_path = os.path.join(separated_folder, vocal_file)
    refined_vocal_path = os.path.join(
        output_dir, f"{os.path.splitext(os.path.basename(input_file))[0]}_refined.wav")

    if os.path.exists(src_vocal_path):
        shutil.move(src_vocal_path, refined_vocal_path)
        print(f"Refined vocal file saved: {refined_vocal_path}")
    else:
        print(f"File not found: {vocal_file}")
        refined_vocal_path = None

    other_file = "other.wav"
    src_other_path = os.path.join(separated_folder, other_file)
    refined_other_path = os.path.join(
        output_dir, f"{os.path.splitext(os.path.basename(input_file))[0]}_back_music.wav")

    if os.path.exists(src_other_path):
        shutil.move(src_other_path, refined_other_path)
        print(f"Refined other file saved: {refined_other_path}")
    else:
        print(f"File not found: {other_file}")
        refined_other_path = None

    htdemucs_dir = os.path.join(output_dir, "htdemucs")
    if os.path.exists(htdemucs_dir):
        shutil.rmtree(htdemucs_dir)
        print(f"Deleted htdemucs directory: {htdemucs_dir}")

    return refined_vocal_path, refined_other_path


def extract_audio(video_path, output_path):
    video = VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile(output_path)
    video.close()
    audio.close()


def generate_transcript(refined_vocal_path, output_folder):
    client = genai.Client(api_key="AIzaSyC4pQM5Tv0T9UxjFWfC9kM2V460HlM7j6Q")

    myfile = client.files.upload(file=refined_vocal_path)

    response = client.models.generate_content(
        model='gemini-2.0-flash',
        contents=['Please transcribe the audio file', myfile]
    )

    response1 = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=f"translate this to english {response.text} only 1 response",
    )

    transcript_text = f"Tamil :{response.text}\nEnglish :{response1.text}"

    transcript_filename = os.path.join(
        output_folder, f"{os.path.basename(refined_vocal_path).split('.')[0]}_transcript.txt")

    os.makedirs(output_folder, exist_ok=True)

    with open(transcript_filename, 'w') as f:
        f.write(transcript_text)

    print(f"Transcript saved: {transcript_filename}")


def perform_diarization(audio_path):
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1", use_auth_token="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")
    diarization = pipeline(audio_path)
    speaker_segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segment = {
            'speaker': speaker,
            'start': turn.start,
            'end': turn.end
        }
        speaker_segments.append(segment)

    return speaker_segments


def cut_segments(video_path, audio_path, speaker_segments, output_folder="segments"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    video = VideoFileClip(video_path)
    audio = AudioFileClip(audio_path)

    try:
        combined_segments = []
        current_segment = speaker_segments[0]

        for i in range(1, len(speaker_segments)):
            if speaker_segments[i]['speaker'] == current_segment['speaker']:
                current_segment['end'] = speaker_segments[i]['end']
            else:
                combined_segments.append(current_segment)
                current_segment = speaker_segments[i]
        combined_segments.append(current_segment)

        for i, segment in enumerate(combined_segments):
            start_time = segment['start']
            end_time = segment['end']
            speaker = segment['speaker']

            video_segment = video.subclipped(start_time, end_time)
            video_output_path = os.path.join(
                output_folder, f"{speaker}_segment_{i}.mp4")
            video_segment.write_videofile(video_output_path)

            audio_segment = audio.subclipped(start_time, end_time)
            audio_output_path = os.path.join(
                output_folder, f"{speaker}_segment_{i}.wav")
            audio_segment.write_audiofile(audio_output_path)

            refined_vocal_path, refined_other_path = separate_vocals_and_other(
                audio_output_path, output_dir=output_folder)
            if refined_vocal_path:
                print(f"Refined vocal file saved: {refined_vocal_path}")
                generate_transcript(refined_vocal_path, output_folder)
            if refined_other_path:
                print(f"Refined back music file saved: {refined_other_path}")
    finally:
        video.close()
        audio.close()


if __name__ == "__main__":
    video_file = "samplevv.mp4"
    audio_file = "output_audio.wav"
    try:
        extract_audio(video_file, audio_file)
        speaker_segments = perform_diarization(audio_file)
        print("Speaker segments:", speaker_segments)
        cut_segments(video_file, audio_file, speaker_segments,
                     output_folder="segments")
    except Exception as e:
        print(f"Error: {e}")
