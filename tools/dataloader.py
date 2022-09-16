import cv2
from decord import VideoReader
from decord import cpu as dcpu


class CLoadVideo:  # for inference
    def __init__(self, path, img_size=(1088, 608), play_seconds=None,
                 vid_sec_to_play=None, allseconds=False, decord=False):
        """
        inputs:
           path     - path to the video
           img_size - the size of image fed to the CNN, must be modulo 32

        self.width, self.height are the resolution of the output video
        """
        print(f'Loading video {path} ...')
        self.decord = decord
        if decord:
            self.vr = VideoReader(path, ctx=dcpu(0))
            self.frame_rate = int(round(self.vr.get_avg_fps()))
            self.vid_num_frames = len(self.vr)
            a_frame = self.vr.next()
            self.vh, self.vw, _ = a_frame.shape
        else:
            self.cap = cv2.VideoCapture(path)
            self.frame_rate = int(round(self.cap.get(cv2.CAP_PROP_FPS)))
            self.vw = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.vh = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.vid_num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f'Opened video {path} @ {self.frame_rate} with '
              'cv2.VideoCapture reader ...')
        print('Lenth of the video: {:d} frames'.format(self.vid_num_frames))
        print(f'Framerate: {self.frame_rate}')

        if allseconds:
            self.play_seconds = [i for i in range(self.vid_num_frames // self.frame_rate)]
            self.vid_sec_to_play = {i: 0 for i in range(self.vid_num_frames // self.frame_rate)}
        else:
            self.play_seconds = play_seconds
            self.vid_sec_to_play = vid_sec_to_play
        self.play_frames = int(len(self.play_seconds) * self.frame_rate)

        # This image size must be divisible by 32
        if img_size == -1:
            self.width = self.vw
            self.height = self.vh
        else:
            self.width = img_size[0]
            self.height = img_size[1]

        # This is the video frame number:
        self.vid_fnum = 0
        # This is the number of frames we've yielded:
        self.yielded_frames = 0

        # The output video size
        # self.w, self.h = 1920, 1080
        self.w, self.h = self.vw, self.vh

    def read_frame(self):
        if self.decord:
            return None, self.vr.next()
        else:
            return self.cap.read()

    def get_a_frame(self):
        '''
        read a frame from the video stream
        '''
        _, img0 = self.read_frame()  # self.cap.read()
        assert img0 is not None, f'Failed to load frame {self.vid_fnum}'
        return img0

    def __iter__(self):
        self.vid_fnum = 0
        self.yielded_frames = 0
        return self

    def get_true_fnum(self, yielded_frames):
        '''
        Convert self.yielded_frames into real fnum
        '''
        second_tbl_idx = int(yielded_frames // self.frame_rate)
        vid_second = self.play_seconds[second_tbl_idx]
        remainder = yielded_frames % self.frame_rate
        vid_fnum = int(vid_second * self.frame_rate) + remainder
        return vid_fnum

    def __next__(self):
        '''
        self.vid_fnum - the iterator index for this object
        '''
        if self.yielded_frames == self.play_frames:
            raise StopIteration

        goal_vid_fnum = self.get_true_fnum(self.yielded_frames)

        if goal_vid_fnum != self.vid_fnum:
            delta = goal_vid_fnum - self.vid_fnum - 1
            assert goal_vid_fnum > self.vid_fnum
            for i in range(delta):
                img = self.get_a_frame()
                self.vid_fnum += 1

        img = self.get_a_frame()
        img = cv2.resize(img, (self.width, self.height))
        self.vid_fnum += 1

        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        vid_sec = int(goal_vid_fnum // self.frame_rate)
        play_num = self.vid_sec_to_play[vid_sec]

        self.yielded_frames += 1
        return self.vid_fnum, play_num, img

    def __len__(self):
        return self.play_seconds * self.frame_rate
