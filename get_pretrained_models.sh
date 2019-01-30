# Create and go to data/
mkdir saves/
cd saves/

# Download pretrained models from Google Drive
# Only TDC and CMC are needed for cycle consistency, t-SNE, and reinforcement learning
# curl -L -o best_cmc_cls.pth "https://drive.google.com/uc?export=download&id=1Qj38BiPIJc91tiQjWoYpjx1cYOGkxayB"
curl -L -o best_cmc.pth "https://drive.google.com/uc?export=download&id=1Sz9wv-bOR58mUQTDFNA7GEKNSMBfFuiC"
# curl -L -o best_optim.pth "https://drive.google.com/uc?export=download&id=1IzVILTYj7Bajfk6pASVkAye3LPLo9CRS"
# curl -L -o best_tdc_cls.pth "https://drive.google.com/uc?export=download&id=1-sWjNH4wEG-BnJy5EEWKsrz4M9JLicZk"
curl -L -o best_tdc.pth "https://drive.google.com/uc?export=download&id=195IUke66agGmVUA_oJ5zD6lNn4_MqdYc"
