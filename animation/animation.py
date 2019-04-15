import matplotlib.animation as animation

def pause(i,j,k):
    pass

class Movie(object):
    def __init__(self, fig, func):
        self.fig = fig
        self.fps = func


class Action(object):
    def __init__(self, fig, loop_fun, start_time, end_time, fargs={}, in_progress=1, title='action'):
        self.title = title
        if loop_fun is not None:
            self.loop_fun = loop_fun
        else:
            self.loop_fun = pause
        self.start_time = start_time
        self.end_time = end_time
        self.in_progress = in_progress
        self.fargs = fargs
        self.fig = fig

    def time_to_frame(self, fps=30):
        self.start_frame = int(self.start_time * float(fps))
        self.end_frame = int(self.end_time * float(fps)) - 1
        self.nframes = self.end_frame - self.start_frame + 1

    def run(self, i):
        if i >= self.start_frame and i <= self.end_frame:
            j = i - self.start_frame
            self.loop_fun(j, self.nframes, self.fargs)
        if i == self.end_frame:
            self.in_progress = 0
        return self.in_progress


class Scene(object):
    def __init__(self, fig, actions=[], fps=30, file_name='test.mp4'):
        self.fig = fig
        self.fps = fps
        self.actions = actions
        self.file_name = file_name

    def add_actions(self, actions):
        self.actions += actions

    def run(self, dpi=300, file_name='sample.mp4', bitrate=-1, times=None):
        writer = animation.FFMpegWriter(fps=self.fps, bitrate=bitrate, codec="libx264")
        for action in self.actions:
            action.time_to_frame(self.fps)
            action.fig = self.fig
            action.in_progress = 1

        in_progress = 1
        frame = 0
        with writer.saving(self.fig, file_name, dpi):
            while in_progress:
                in_progress = 0
                for action in self.actions:
                    in_progress += action.run(frame)
                frame += 1
                if (times is None) or (frame >= times[0] * self.fps and frame <= times[1] * self.fps):
                    self.fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
                    writer.grab_frame(facecolor=self.fig.get_facecolor())