import imageio
import numpy as np
from PIL import Image
from tqdm import tqdm


class VideoData:
    def __init__(self, video_file, height=None, width=None):
        if video_file is None:
            raise ValueError("`video_file` is required.")
        self.video_file = str(video_file)
        self.reader = imageio.get_reader(video_file)
        self.length = None
        self.set_shape(height, width)

    def raw_data(self):
        return [self[index] for index in range(len(self))]

    def set_length(self, length):
        self.length = length

    def set_shape(self, height, width):
        self.height = height
        self.width = width

    def __len__(self):
        if self.length is None:
            return self.reader.count_frames()
        return self.length

    def shape(self):
        if self.height is not None and self.width is not None:
            return self.height, self.width
        width, height = self[0].size
        return height, width

    def center_crop_and_resize(self, image, height, width):
        image = np.array(image)
        image_height, image_width, _ = image.shape
        if image_height / image_width < height / width:
            cropped_width = int(image_height / height * width)
            left = (image_width - cropped_width) // 2
            image = image[:, left : left + cropped_width]
            image = Image.fromarray(image).resize((width, height))
        else:
            cropped_height = int(image_width / width * height)
            top = (image_height - cropped_height) // 2
            image = image[top : top + cropped_height, :]
            image = Image.fromarray(image).resize((width, height))
        return image

    def __getitem__(self, item):
        frame = Image.fromarray(np.array(self.reader.get_data(item))).convert("RGB")
        width, height = frame.size
        if self.height is not None and self.width is not None:
            if self.height != height or self.width != width:
                frame = self.center_crop_and_resize(frame, self.height, self.width)
        return frame

    def __del__(self):
        if hasattr(self, "reader"):
            self.reader.close()


def save_video(frames, save_path, fps, video_encoding_quality=9, ffmpeg_params=None):
    writer = imageio.get_writer(
        save_path,
        fps=fps,
        quality=video_encoding_quality,
        ffmpeg_params=ffmpeg_params,
    )
    for frame in tqdm(frames, desc="Saving video"):
        writer.append_data(np.array(frame))
    writer.close()
