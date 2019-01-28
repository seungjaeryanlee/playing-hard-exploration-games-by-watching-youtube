# Create and go to data/
mkdir data/
cd data/

# Download videos from Google Drive
curl -L -o 2AYaxTiWKoY.mp4 "https://drive.google.com/uc?export=download&id=1d96v8R85Hz0_73zvb78aHIK1cntt8W-4"
curl -L -o 6zXXZvVvTFs.mp4 "https://drive.google.com/uc?export=download&id=1M_kIL-Xmdw-xlf_f6HrF94tDdcS3St8J"
curl -L -o pF6xCZA72o0.mp4 "https://drive.google.com/uc?export=download&id=18tmjd0w7_WprPnEhOEWIpgXxqrbey2mb"
curl -L -o SuZVyOlgVek.mp4 "https://drive.google.com/uc?export=download&id=1rK6h9Ya-3mpNoDAGwR2GYfiVavAv7rNw"
curl -L -o sYbBgkP9aMo.mp4 "https://drive.google.com/uc?export=download&id=18Zhkx4opv7lvpoPBm-L95RSkA_JuHoD8"

# Extract audio
ffmpeg -i 2AYaxTiWKoY.mp4 2AYaxTiWKoY.wav
ffmpeg -i 6zXXZvVvTFs.mp4 6zXXZvVvTFs.wav
ffmpeg -i pF6xCZA72o0.mp4 pF6xCZA72o0.wav
ffmpeg -i SuZVyOlgVek.mp4 SuZVyOlgVek.wav
ffmpeg -i sYbBgkP9aMo.mp4 sYbBgkP9aMo.wav
