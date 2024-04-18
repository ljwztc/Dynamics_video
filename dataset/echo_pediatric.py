"""EchoNet-Dynamic Dataset."""

import math
import os
import collections
import pandas

import cv2

import numpy as np
import skimage.draw
from torch.utils.data import Dataset, DataLoader


class Echo_pediatric_seg(Dataset):
    """EchoNet-Dynamic Dataset.

    Args:
        root (string): Root directory of dataset (defaults to `echonet.config.DATA_DIR`)
        split (string): One of {0~9, ``all''}
        target_type (string or list, optional): Type of target to use,
            ``Filename'', ``EF'', ``EDV'', ``ESV'', ``LargeIndex'',
            ``SmallIndex'', ``LargeFrame'', ``SmallFrame'', ``LargeTrace'',
            or ``SmallTrace''
            Can also be a list to output a tuple with all specified target types.
            The targets represent:
                ``Filename'' (string): filename of video
                ``EF'' (float): ejection fraction
                ``EDV'' (float): end-diastolic volume
                ``ESV'' (float): end-systolic volume
                ``LargeIndex'' (int): index of large (diastolic) frame in video
                ``SmallIndex'' (int): index of small (systolic) frame in video
                ``LargeFrame'' (np.array shape=(3, height, width)): normalized large (diastolic) frame
                ``SmallFrame'' (np.array shape=(3, height, width)): normalized small (systolic) frame
                ``LargeTrace'' (np.array shape=(height, width)): left ventricle large (diastolic) segmentation
                    value of 0 indicates pixel is outside left ventricle
                             1 indicates pixel is inside left ventricle
                ``SmallTrace'' (np.array shape=(height, width)): left ventricle small (systolic) segmentation
                    value of 0 indicates pixel is outside left ventricle
                             1 indicates pixel is inside left ventricle
            Defaults to ``EF''.
        mean (int, float, or np.array shape=(3,), optional): means for all (if scalar) or each (if np.array) channel.
            Used for normalizing the video. Defaults to 0 (video is not shifted).
        std (int, float, or np.array shape=(3,), optional): standard deviation for all (if scalar) or each (if np.array) channel.
            Used for normalizing the video. Defaults to 0 (video is not scaled).
        length (int or None, optional): Number of frames to clip from video. If ``None'', longest possible clip is returned.
            Defaults to 16.
        period (int, optional): Sampling period for taking a clip from the video (i.e. every ``period''-th frame is taken)
            Defaults to 2.
        max_length (int or None, optional): Maximum number of frames to clip from video (main use is for shortening excessively
            long videos when ``length'' is set to None). If ``None'', shortening is not applied to any video.
            Defaults to 250.
        clips (int, optional): Number of clips to sample. Main use is for test-time augmentation with random clips.
            Defaults to 1.
        pad (int or None, optional): Number of pixels to pad all frames on each side (used as augmentation).
            and a window of the original size is taken. If ``None'', no padding occurs.
            Defaults to ``None''.
        noise (float or None, optional): Fraction of pixels to black out as simulated noise. If ``None'', no simulated noise is added.
            Defaults to ``None''.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        external_test_location (string): Path to videos to use for external testing.
    """

    def __init__(self, root,
                 split="train", target_type="EF",
                 mean=0., std=1.,
                 length=1, period=1,
                 max_length=250,
                 validation = True,
                 clips=1,
                 max_clips=100,
                 pad=None,
                 grey=True,
                 noise=None,
                 target_transform=None,
                 external_test_location=None):
        self.root = root
        self.split = split
        if not isinstance(target_type, list):
            target_type = [target_type]
        self.target_type = target_type
        self.validation = validation
        self.mean = mean
        self.std = std
        self.length = length
        self.max_length = max_length
        self.period = period
        self.clips = clips
        self.max_clips = max_clips
        self.pad = pad
        self.grey = grey
        self.noise = noise
        self.target_transform = target_transform
        self.external_test_location = external_test_location

        self.fnames, self.outcome = [], []

        if self.split == "EXTERNAL_TEST":
            self.fnames = sorted(os.listdir(self.external_test_location))
        else:
            # Load video-level labels
            with open(os.path.join(self.root, "FileList.csv")) as f:
                data = pandas.read_csv(f)
            # data["Split"].map(lambda x: x.upper())

            if self.split != "ALL":
                data = data[data["Split"] == self.split]

            self.header = data.columns.tolist()
            self.fnames = data["FileName"].tolist()
            self.fnames = [fn if os.path.splitext(fn)[1] != "" else fn + ".avi" for fn in self.fnames]  # Assume avi if no suffix
            self.outcome = data.values.tolist()

            # Check that files are present
            # TODO: should just make this work on nexted dir
            # missing = set(self.fnames) - set(os.listdir(os.path.join(self.root, "Videos")))
            # if len(missing) != 0:
            #     print("{} videos could not be found in {}:".format(len(missing), os.path.join(self.root, "Videos")))
            #     for f in sorted(missing):
            #         print("\t", f)
            #     raise FileNotFoundError(os.path.join(self.root, "Videos", sorted(missing)[0]))

            # Load traces
            self.frames = collections.defaultdict(list)
            self.trace = collections.defaultdict(_defaultdict_of_lists)

            with open(os.path.join(self.root, "VolumeTracings.csv")) as f:
                header = f.readline().strip().split(",")
                if header == ["FileName", "X1", "Y1", "X2", "Y2", "Frame"]:
                    for line in f:
                        filename, x1, y1, x2, y2, frame = line.strip().split(',')
                        x1 = float(x1)
                        y1 = float(y1)
                        x2 = float(x2)
                        y2 = float(y2)
                        frame = int(frame)
                        if frame not in self.trace[filename]:
                            self.frames[filename].append(frame)
                        self.trace[filename][frame].append((x1, y1, x2, y2))
                if header == ["FileName", "X", "Y", "Frame"]:
                    # TODO: probably could merge
                    for line in f:
                        filename, x, y, frame = line.strip().split(',')
                        if frame in ["No Systolic", "No Diastolic"]:
                            self.frames[filename].append(None)
                        else:
                            frame = int(frame)
                            x = float(x)
                            y = float(y)
                            if frame not in self.trace[filename]:
                                self.frames[filename].append(frame)
                            self.trace[filename][frame].append((x, y))
            for filename in self.frames:
                for frame in self.frames[filename]:
                    self.trace[filename][frame] = np.array(self.trace[filename][frame])

            # A small number of videos are missing traces; remove these videos
            # keep = [len(self.frames[f]) >= 2 for f in self.fnames]
            # self.fnames = [f for (f, k) in zip(self.fnames, keep) if k]
            # self.outcome = [f for (f, k) in zip(self.outcome, keep) if k]

    def __getitem__(self, index):
        # Find filename of video
        if self.split == "EXTERNAL_TEST":
            video = os.path.join(self.external_test_location, self.fnames[index])
        elif self.split == "CLINICAL_TEST":
            video = os.path.join(self.root, "ProcessedStrainStudyA4c", self.fnames[index])
        else:
            video = os.path.join(self.root, "Videos", self.fnames[index])

        # Load video into np.array
        video, fps = loadvideo(video, self.grey)
        video = video.astype(np.float32)
        print(video.shape, fps)

        # Add simulated noise (black out random pixels)
        # 0 represents black at this point (video has not been normalized yet)
        if self.noise is not None:
            n = video.shape[1] * video.shape[2] * video.shape[3]
            ind = np.random.choice(n, round(self.noise * n), replace=False)
            f = ind % video.shape[1]
            ind //= video.shape[1]
            i = ind % video.shape[2]
            ind //= video.shape[2]
            j = ind
            video[:, f, i, j] = 0

        # Apply normalization
        if isinstance(self.mean, (float, int)):
            video -= self.mean
        else:
            video -= self.mean.reshape(3, 1, 1, 1)

        if isinstance(self.std, (float, int)):
            video /= self.std
        else:
            video /= self.std.reshape(3, 1, 1, 1)

        # Set number of frames
        c, f, h, w = video.shape
        if self.length is None:
            # Take as many frames as possible
            length = f // self.period
        else:
            # Take specified number of frames
            length = self.length

        if self.max_length is not None:
            # Shorten videos to max_length
            length = min(length, self.max_length)

        if f < length * self.period:
            # Pad video with frames filled with zeros if too short
            # 0 represents the mean color (dark grey), since this is after normalization
            video = np.concatenate((video, np.zeros((c, length * self.period - f, h, w), video.dtype)), axis=1)
            c, f, h, w = video.shape  # pylint: disable=E0633

        if self.clips == "all":
            # Take all possible clips of desired length
            start = np.arange(f - (length - 1) * self.period)
            if start.size > self.max_clips:
                # TODO: this messes up the clip number in test-time aug
                # Might need to have a clip index target
                start = np.random.choice(start, self.max_clips, replace=False)
                start.sort()
        else:
            # Take random clips from video
            start = np.random.choice(f - (length - 1) * self.period, self.clips)

        # Gather targets
        target = []
        for t in self.target_type:
            key = self.fnames[index]
            if t == "Filename":
                target.append(self.fnames[index])
            elif t == "LargeIndex":
                # Traces are sorted by cross-sectional area
                # Largest (diastolic) frame is last
                target.append(np.int64(self.frames[key][-1]))
            elif t == "SmallIndex":
                # Largest (diastolic) frame is first
                target.append(np.int64(self.frames[key][0]))
            elif t in ["LargeFrame", "SmallFrame"]:
                if t == "LargeFrame":
                    frame = self.frames[key][-1]
                else:
                    frame = self.frames[key][0]

                if frame is None or frame >= video.shape[1]:
                    target.append(np.full((video.shape[0], video.shape[2], video.shape[3]), math.nan, video.dtype))
                else:
                    target.append(video[:, frame, :, :])
            elif t in ["LargeTrace", "SmallTrace"]:
                if t == "LargeTrace":
                    frame = self.frames[key][-1]
                else:
                    frame = self.frames[key][0]
                if frame is None or frame >= video.shape[1]:
                    mask = np.full((video.shape[2], video.shape[3]), math.nan, np.float32)
                else:
                    t = self.trace[key][frame]

                    if t.shape[1] == 4:
                        x1, y1, x2, y2 = t[:, 0], t[:, 1], t[:, 2], t[:, 3]
                        x = np.concatenate((x1[1:], np.flip(x2[1:])))
                        y = np.concatenate((y1[1:], np.flip(y2[1:])))
                    else:
                        assert t.shape[1] == 2
                        x, y = t[:, 0], t[:, 1]

                    r, c = skimage.draw.polygon(np.rint(y).astype(np.int64), np.rint(x).astype(np.int64), (video.shape[2], video.shape[3]))
                    mask = np.zeros((video.shape[2], video.shape[3]), np.float32)
                    mask[r, c] = 1
                target.append(mask)
            else:
                if self.split == "CLINICAL_TEST" or self.split == "EXTERNAL_TEST":
                    target.append(np.float32(0))
                else:
                    # target.append(np.float32(self.outcome[index][self.header.index(t)]))  # TODO: is floating necessary
                    target.append(self.outcome[index][self.header.index(t)])

        if target != []:
            target = tuple(target) if len(target) > 1 else target[0]
            if self.target_transform is not None:
                target = self.target_transform(target)

        # Select clips from video
        video = tuple(video[:, s + self.period * np.arange(length), :, :] for s in start)
        if self.clips == 1:
            video = video[0]
        else:
            video = np.stack(video)


        return video, target

    def __len__(self):
        return len(self.fnames)

    def extra_repr(self) -> str:
        """Additional information to add at end of __repr__."""
        lines = ["Target type: {target_type}", "Split: {split}"]
        return '\n'.join(lines).format(**self.__dict__)


def loadvideo(filename: str, grey:bool) -> np.ndarray:
    """Loads a video from a file.
    Args:
        filename (str): filename of video
    Returns:
        A np.ndarray with dimensions (channels=3, frames, height, width). The
        values will be uint8's ranging from 0 to 255.
    Raises:
        FileNotFoundError: Could not find `filename`
        ValueError: An error occurred while reading the video
    """

    if not os.path.exists(filename):
        raise FileNotFoundError(filename)
    capture = cv2.VideoCapture(filename)

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps = int(capture.get(cv2.CAP_PROP_FPS))

    v = np.zeros((frame_count, frame_height, frame_width, 1 if grey else 3), np.uint8)

    for count in range(frame_count):
        ret, frame = capture.read()
        if not ret:
            raise ValueError("Failed to load frame #{} of {}.".format(count, filename))
        if grey:
            frame = np.expand_dims(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), axis=-1)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        v[count, ...] = frame

    v = v.transpose((3, 0, 1, 2))

    return v, fps

def _defaultdict_of_lists():
    """Returns a defaultdict of lists.

    This is used to avoid issues with Windows (if this function is anonymous,
    the Echo dataset cannot be used in a dataloader).
    """

    return collections.defaultdict(list)


if __name__ == '__main__':
    data_loader = Echo_pediatric_seg(
            root='/data/liujie/data/echocardiogram/pediatric_echo_avi/A4C', split='ALL', target_type="SmallTrace"
        )
    train_loader = DataLoader(data_loader, batch_size=1, shuffle=False, num_workers=1)
    print(data_loader)
    for targets, targets_gt in train_loader:
        pass
        # print(targets.shape, targets_gt.shape)

        # import numpy as np
        # import matplotlib.pyplot as plt

        # image = targets.squeeze()
        # mask = targets_gt.squeeze()

        # fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # # 可视化图像
        # axes[0].imshow(image, cmap='gray')
        # axes[0].set_title('Image')
        # axes[0].axis('off')

        # axes[1].imshow(mask, cmap='gray')
        # axes[1].set_title('Mask')
        # axes[1].axis('off')

        # axes[2].imshow(image, cmap='gray')
        # axes[2].imshow(mask, cmap='Reds', alpha=0.5)
        # axes[2].set_title('Image with Mask')
        # axes[2].axis('off')

        # fig.subplots_adjust(wspace=0.3)

        # # 保存图像为PNG文件
        # fig.savefig('image_mask_visualization.png', dpi=300, bbox_inches='tight')
        # exit()