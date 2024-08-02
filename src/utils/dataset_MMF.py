import os
import numpy as np
import librosa
import pandas as pd
import pickle

def extract_mfcc(file_path, n_mfcc=259):
    y, sr = librosa.load(file_path)
    stft = np.abs(librosa.stft(y))
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(stft), sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc, axis=1)

def process_audio_folder(folder_path):
    audio_features = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(".wav") and filename.startswith("01-"):
            file_path = os.path.join(folder_path, filename)
            mfcc = extract_mfcc(file_path)
            audio_features[filename] = mfcc
            print(f"Processed {filename} and extracted MFCCs")
    return audio_features

def get_file_paths(root_dir):
    file_paths = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths

def extract_facial_features(video_path, output_dir, openface_path):
    command = f'{openface_path} -au_static -f {video_path} -out_dir {output_dir}'
    os.system(command)

def process_video_folder(vid_folder, openface_path):
    cols_to_select = [
        'AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 'AU09_r', 'AU10_r',
        'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r',
        'AU45_r', 'AU01_c', 'AU02_c', 'AU04_c', 'AU05_c', 'AU06_c', 'AU07_c', 'AU09_c',
        'AU10_c', 'AU12_c', 'AU14_c', 'AU15_c', 'AU17_c', 'AU20_c', 'AU23_c', 'AU25_c',
        'AU26_c', 'AU28_c', 'AU45_c'
    ]

    video_features = {}
    video_paths = get_file_paths(vid_folder)

    for video_path in video_paths:
        video_filename = os.path.basename(video_path)
        if video_filename.startswith("01-"):
            print(f"Processing video: {video_path}")
            actor_name = os.path.basename(os.path.dirname(video_path))
            video_name = os.path.splitext(video_filename)[0]
            output_dir = os.path.join('./outs', actor_name, video_name)
            os.makedirs(output_dir, exist_ok=True)
            extract_facial_features(video_path, output_dir, openface_path)

            all_features_file = os.path.join(output_dir, f'{video_name}.csv')
            if not os.path.isfile(all_features_file):
                print(f"Error: File {all_features_file} not found.")
                continue

            features_df = pd.read_csv(all_features_file)
            au_features_df = features_df[cols_to_select].mean(axis=0)
            video_features[video_filename] = au_features_df.values
            print(f"Extracted facial action units for {video_name}")

    return video_features

def main():
    audio_folder_path = input("Enter the folder path containing the WAV files: ")
    video_folder_path = input("Enter the folder path containing the video files: ")
    openface_path = input("Enter the path to the OpenFace FeatureExtraction binary: ")
    output_pickle_file = input("Enter the path for the output pickle file (e.g., au_mfcc.pkl): ")

    audio_features = process_audio_folder(audio_folder_path)
    video_features = process_video_folder(video_folder_path, openface_path)

    combined_features = {}

    for key in audio_features.keys():
        if key in video_features:
            modality, vocal_channel, emotion, intensity, statement, repetition, actor = key.split('-')
            key_formatted = f'{modality}-{vocal_channel}-{emotion}-{intensity}-{statement}-{repetition}-{actor}'

            combined_array = np.zeros(294)
            combined_array[:35] = video_features[key]
            combined_array[35:] = audio_features[key]

            combined_features[key_formatted] = combined_array

    with open(output_pickle_file, 'wb') as f:
        pickle.dump(combined_features, f)

    print(f"Saved combined features to {output_pickle_file}")

if __name__ == "__main__":
    main()