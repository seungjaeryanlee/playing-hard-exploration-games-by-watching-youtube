"""Lazy version of the dataset for training TDC and CMC."""
import cv2
import librosa
import numpy as np
import torch
from skvideo.io import FFmpegReader
from torch.utils.data import Dataset


class LazyTDCCMCDataset(Dataset):
    """
    Dataset for training TDC and CMC.

    Dataset for sampling video frames and audio snippets with distance
    labels to train the embedding networks.

    Parameters
    ----------
    filenames : list of str
        List of filenames of video files.
    trims : list of float
        List of tuples `(begin_idx, end_idx)` that specify what frame
        of the video to start and end.
    crops : list of tuple
        List of tuples `(x_1, y_1, x_2, y_2)` that define the clip
        window of each video.
    frame_rate : int
        Frame rate to sample video. Default to 15.

    """

    def __init__(self, filenames, trims, crops, frame_rate=15):
        # TDCCMCDataset is an unconvential dataset, where each data is
        # dynamically sampled whenever needed instead of a static dataset.
        # Therefore, in `__init__`, we do not define a static dataset. Instead,
        # we simply preprocess the video and audio for faster `__getitem__`.

        super().__init__()
        self.filenames = filenames
        self.trims = trims
        self.crops = crops

        self.audios = []
        self.readers = []
        for filename in filenames:
            # Get video frames with scikit-video
            reader = FFmpegReader(
                filename + ".mp4",
                inputdict={"-r": str(frame_rate)},
                outputdict={"-r": str(frame_rate)},
            )
            self.readers.append(reader)

            # STFT audio
            # TODO Magic number sr=2000, n_fft=510
            y, _ = librosa.load(filename + ".wav", sr=2000)
            D = librosa.core.stft(y, n_fft=510)
            D = np.abs(D)

            # Save audio
            self.audios.append(D)

    def __len__(self):
        # Return a high number since this dataset in dynamic. Don't use
        # this explicitly!
        return np.iinfo(np.int64).max

    def __getitem__(self, index):
        """
        Return a sample from the dynamic dataset.

        Each sample contains two video frames, one audio snippet, one
        TDC label and one CMC label. In other words, the format is
        (frame_v, frame_w, audio_a, tdc_label, cmc_label).

        Parameters
        ----------
        index : int

        Returns
        -------
        frame_v : torch.FloatTensor
        frame_w : torch.FloatTensor
        audio_a
        tdc_label : torch.LongTensor
        cmc_label : torch.LongTensor

        """
        # Below is a paragraph from the original paper:
        #
        # To generate training data, we sample input pairs (v^i, w^i) (where
        # v^i and w^i are sampled from the same domain) as follows. First, we
        # sample a demonstration sequence from our three training videos. Next,
        # we sample both an interval, d_k ∈ {[0], [1], [2], [3 - 4], [5 - 20],
        # [21 - 200]}, and a distance, ∆t ∈ dk. Finally, we randomly select a
        # pair of frames from the sequence with temporal distance ∆t. The model
        # is trained with Adam using a learning rate of 10^-4 and batch size of
        # 32 for 200,000 steps.
        #
        # From Section 5: Implementation Details

        # 1) Sample video from videos
        src_idx = np.random.choice(len(self.audios))
        reader = self.readers[src_idx]
        audio = self.audios[src_idx]
        trim = self.trims[src_idx]
        crop = self.crops[src_idx]

        # 2) Sample tdc_label and cmc_label
        tdc_label = self._sample_label()
        cmc_label = self._sample_label()

        # 3) Sample tdc_distance, cmc_distance
        tdc_distance = self._sample_distance_from_label(tdc_label)
        cmc_distance = self._sample_distance_from_label(cmc_label)

        # 4) Sample framestack_v from video (check limits carefully)
        framestack_v_idx = np.random.randint(0, reader.getShape()[0] - tdc_distance - 4)
        framestack_v = self._sample_framestack(framestack_v_idx, reader, trim, crop)

        # 5) Sample frame_w from video
        framestack_w_idx = framestack_v_idx + tdc_distance
        framestack_w = self._sample_framestack(framestack_w_idx, reader, trim, crop)

        # 6) Sample audio_a from audio
        audio_a_idx = framestack_v_idx + cmc_distance
        audio_a = audio[:, audio_a_idx : audio_a_idx + 137]
        audio_a = torch.FloatTensor(audio_a)

        # 7) Crop Frames from 140x140 to 128x128
        # TODO Is it correct to use same crop for both v and w?
        y = np.random.randint(0, 140 - 128)
        x = np.random.randint(0, 140 - 128)
        framestack_v = framestack_v[:, :, y : y + 128, x : x + 128]
        framestack_w = framestack_w[:, :, y : y + 128, x : x + 128]

        # 8) Switch 4 x 3 x 128 x 128 to 1 x 12 x 128 x 128
        framestack_v = torch.FloatTensor(framestack_v).view(-1, 128, 128)
        framestack_w = torch.FloatTensor(framestack_w).view(-1, 128, 128)

        # 9) Scale image values from 0~255 to 0~1
        framestack_v /= 255.0
        framestack_w /= 255.0

        # 10) Return (frame_v, frame_w, audio_a, tdc_label, cmc_label)
        return (
            framestack_v,
            framestack_w,
            audio_a,
            torch.LongTensor([tdc_label]),
            torch.LongTensor([cmc_label]),
        )

    def _sample_label(self):
        """
        Sample randomly from label.

        Returns
        -------
        label : int
            Label sampled from 0 ~ 5.

        """
        return np.random.choice(6)

    def _sample_distance_from_label(self, label):
        """
        Sample randomly from distance from label.

        Label 0: Distance 0
        Label 1: Distance 1
        Label 2: Distance 2
        Label 3: Distance sampled from [3, 4]
        Label 4: Distance sampled from [5, 20]
        Label 5: Distance sampled from [21, 200]

        Parameters
        ----------
        label : int
            Label sampled randomly.

        Returns
        -------
        distance: int
            Distance sampled according to the label.

        """
        if label == 0:  # [0]
            distance = 0
        elif label == 1:  # [1]
            distance = 1
        elif label == 2:  # [2]
            distance = 2
        elif label == 3:  # [3 - 4]
            distance = np.random.choice(np.arange(3, 4 + 1))
        elif label == 4:  # [5 - 20]
            distance = np.random.choice(np.arange(5, 20 + 1))
        else:  # [21 - 200]
            distance = np.random.choice(np.arange(21, 200 + 1))

        return distance

    def _sample_framestack(self, start_frame, reader, trim, crop):
        assert start_frame + trim[0] + 4 < reader.getShape()[0]
        framestack = []
        for frame_idx, frame in enumerate(reader.nextFrame()):
            # Trim video (time)
            if start_frame + trim[0] <= frame_idx < start_frame + trim[0] + 4:
                # Crop frames (space)
                frame = frame[crop[1] : crop[3], crop[0] : crop[2], :]
                framestack.append(cv2.resize(frame, (140, 140)))
            if frame_idx == start_frame + trim[0] + 4:
                break

        # Change to NumPy array with PyTorch dimension format
        framestack = np.array(framestack, dtype=float)
        framestack = np.transpose(framestack, axes=(0, 3, 1, 2))

        return framestack
