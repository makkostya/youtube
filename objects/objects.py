import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as path
import matplotlib.lines as lines
import matplotlib.text as text_obj

import matplotlib.animation as animation
import math
from scipy.stats import norm
import pickle
import io
import time
import gc

def sampling(p1,p2,num,d):
    t = np.linspace(0,1,num)
    t = t ** d
    sol = np.zeros(num)
    for i in range(num):
        sol[i] = p1 + (p2 - p1)*t[i]
    return sol

class Hist(object):
    def __init__(self, axes, bins, values, facecolor='#0165fc', edgecolor='None', alpha=0.85):
        self.bins = bins
        self.values = values
        self.axes = axes
        left = np.array(bins[:-1]) + 0.1 * (bins[1] - bins[0])
        right = np.array(bins[1:]) - 0.1 * (bins[1] - bins[0])
        nrects = len(left)
        bottom = np.zeros(nrects)
        top = bottom + 0.
        nverts = nrects * (1 + 3 + 1)
        verts = np.zeros((nverts, 2))
        codes = np.ones(nverts, int) * path.Path.LINETO
        codes[0::5] = path.Path.MOVETO
        codes[4::5] = path.Path.CLOSEPOLY
        verts[0::5, 0] = left
        verts[0::5, 1] = bottom
        verts[1::5, 0] = left
        verts[1::5, 1] = top
        verts[2::5, 0] = right
        verts[2::5, 1] = top
        verts[3::5, 0] = right
        verts[3::5, 1] = bottom
        self.verts = verts

        barpath = path.Path(self.verts, codes)
        # self.patch = patches.PathPatch(barpath, facecolor='#0165fc',
        #                           edgecolor='#0343df', alpha=0.5)  # <-- add to figure
        self.patch = patches.PathPatch(barpath, facecolor=facecolor,
                                       edgecolor=edgecolor, alpha=alpha)  # <-- add to figure
        self.axes.add_patch(self.patch)

    def action_grawing_uniform(self, i, nframes, fargs):
        frames = np.linspace(0, 1, nframes)
        n, bins = np.histogram(fargs['data'], bins=self.bins)
        top = frames[i] * n
        self.verts[1::5, 1] = top
        self.verts[2::5, 1] = top

    def action_grawing(self, i, nframes, fargs):
        data = fargs['data']
        accel = fargs['accel']
        frames = sampling(0, len(data), nframes, accel)
        n, bins = np.histogram(data[:int(frames[i]) + 1], bins=self.bins)
        top = n
        self.verts[1::5, 1] = top
        self.verts[2::5, 1] = top

    def action_changing_alpha(self, i, nframes, fargs):
        if i == 0:
            fargs['from'] = fargs['from']()
        frames = np.linspace(fargs['from'], fargs['to'], nframes)
        self.patch.set_alpha(frames[i])

    def action_disappearing(self, i, nframes, fargs):
        frames = np.linspace(1, 0, nframes)
        n, bins = np.histogram(fargs['data'], bins=self.bins)
        top = frames[i] * n
        self.verts[1::5, 1] = top
        self.verts[2::5, 1] = top

class Pdf(object):

    def __init__(self, axes, x, mean, std, scale, color='#8f1402', alpha=1, lw=5):
        self.axes = axes
        self.line = lines.Line2D([], [], alpha=alpha, color=color, lw=lw)
        self.axes.add_line(self.line)
        self.x = x
        self.mean = mean
        self.std = std
        self.scale = scale
        self.y = 0
        self._compute_y()

    def get_prob_for_range(self,a,b):
        cdf_a = norm.cdf(a, loc=self.mean, scale=self.std)
        cdf_b = norm.cdf(b, loc=self.mean, scale=self.std)

        if b >= a:
            return cdf_b - cdf_a
        else:
            return - cdf_b + cdf_a

    def _compute_y(self):
        self.y = self.scale * norm.pdf(self.x, self.mean, self.std)

    def action_drawing(self, i, nframes, fargs=None):
        frames = np.where((self.x >= self.axes.get_xlim()[0]) &
                          (self.x <= self.axes.get_xlim()[1]))[0]
        frames = np.linspace(frames[0], frames[-1], nframes)
        self.line.set_data(self.x[int(frames[0]):int(frames[i])+1],
                           self.y[int(frames[0]):int(frames[i])+1])
        if i == nframes - 1:
            self.line.set_data(self.x, self.y)
        # frames = np.linspace(0, len(self.x), nframes)
        # self.line.set_data(self.x[:int(frames[i])+1], self.y[:int(frames[i])+1])

    def action_changing_alpha(self, i, nframes, fargs):
        if i == 0:
            fargs['from'] = fargs['from']()
        frames = np.linspace(fargs['from'], fargs['to'], nframes)
        self.line.set_alpha(frames[i])

    def action_highlight(self, i, nframes, fargs):
        if i == 0:
            fargs['from'] = self.line.get_linewidth()
            fargs['to'] = 2*self.line.get_linewidth()

        frames1 = np.linspace(fargs['from'], fargs['to'], int(nframes/2))
        frames2 = np.linspace(fargs['to'], fargs['from'], nframes - int(nframes / 2))
        frames = np.hstack([frames1, frames2])
        self.line.set_linewidth(frames[i])

    def action_move(self, i, nframes, fargs):
        if i == 0:
            fargs['from'] = self.mean
        frames = np.linspace(fargs['from'], fargs['to'], nframes)
        self.mean = frames[i]
        self._compute_y()
        self.line.set_data(self.x, self.y)

class Point(object):

    def __init__(self, axes, x, y=0, r=0, aspect=1., ratio = 6., facecolor='darkgreen', edgecolor='None', alpha=1):
        self.axes = axes
        self.x = x
        self.y = y
        self.r = r
        self.ratio = ratio
        self.aspect = aspect
        self.point = patches.Ellipse((self.x, self.y), self.r, self.r / self.aspect,
                                     facecolor=facecolor, edgecolor=edgecolor, alpha=alpha, linewidth=2)
        self.point.set_zorder(self.axes.spines.values()[2].get_zorder()+1)
        # self.point.set_clip_on(False)
        self.axes.add_patch(self.point)
        self.center = patches.Ellipse((self.x, self.y), self.r / self.ratio,
                                      (self.r / self.ratio)/ self.aspect, facecolor='#013a4b')
        self.center.set_zorder(self.axes.spines.values()[2].get_zorder() + 2)
        self.axes.add_patch(self.center)

    def action_changing_radius(self, i, nframes, fargs):
        frames = np.linspace(fargs['from_radius'], fargs['to_radius'], nframes)
        self.r = frames[i]
        self.point.width = self.r
        self.point.height = self.r / self.aspect
        self.center.width = self.point.width / self.ratio
        self.center.height = self.point.height / self.ratio

    def action_appearing(self, i, nframes, fargs):
        ratio = fargs['ratio']
        scale = fargs['scale']

        nframes_1 = int(ratio * nframes)
        nframes_2 = nframes - nframes_1
        frames_1 = np.linspace(0, fargs['to_radius'] * scale, nframes_1)
        frames_2 = np.linspace(fargs['to_radius'] * scale, fargs['to_radius'], nframes_2)
        frames = np.hstack([frames_1, frames_2])
        self.point.set_alpha(1.)
        self.r = frames[i]
        self.point.width = self.r
        self.point.height = self.r / self.aspect
        self.center.width = self.point.width / self.ratio
        self.center.height = self.point.height / self.ratio
        self.point.set_alpha(fargs['alpha'])

    def action_disappearing(self, i, nframes, fargs):
        if i == 0:
            fargs['from'] = self.r
        frames = np.linspace(fargs['from'], 0, nframes)
        self.r = frames[i]
        self.point.width = self.r
        self.point.height = self.r / self.aspect
        self.center.width = self.point.width / self.ratio
        self.center.height = self.point.height / self.ratio

    def action_changing_alpha(self, i, nframes, fargs):
        if i == 0:
            fargs['from'] = fargs['from']()
        frames = np.linspace(fargs['from'], fargs['to'], nframes)
        self.point.set_alpha(frames[i])


class Camera(object):

    def __init__(self, axes, ax_ratio, aspect_ratio):
        self.axes = axes
        self.ax_ratio = ax_ratio
        self.aspect_ratio = aspect_ratio
        self.xlim = axes.get_xlim()
        self.y_min = axes.get_ylim()[0]

    def action_changing(self, i, nframes, fargs):
        if i == 0:
            fargs['xlim_from'] = self.axes.get_xlim()
            fargs['y_min_from'] = self.axes.get_ylim()[0]

        x_min = np.linspace(fargs['xlim_from'][0], fargs['xlim_to'][0], nframes)
        x_max = np.linspace(fargs['xlim_from'][1], fargs['xlim_to'][1], nframes)

        y_min = np.linspace(fargs['y_min_from'], fargs['y_min_to'], nframes)

        self.xlim = [x_min[i], x_max[i]]
        self.y_min = y_min[i]
        self.axes.set_xlim(self.xlim, auto=True)
        self.axes.set_xticks(range(int(math.ceil(x_min[i])),int(math.floor(x_max[i]))))
        # delta_x = x_max[i] - x_min[i]
        # y_max = self.ax_ratio * delta_x + self.y_min
        # self.axes.set_ylim([self.y_min, y_max])

    def action_change_alpha_ticks(self, i, nframes, fargs):
        if i == 0:
            fargs['from'] = 1.
            fargs['tick_length'] = 6
        frames_alpha = np.linspace(fargs['from'], fargs['to'], nframes)
        # frames_l = np.linspace(4, 0, nframes)
        # self.axes.xaxis.set_tick_params(length=frames_l[i])
        for l in self.axes.get_xticklabels(which='both'):
            l.set_alpha(frames_alpha[i])
        # self.axes.get_xaxis().set_alpha(frames[i])
        # for t in self.axes.get_xaxis().iter_ticks():
        #     t.set_alpha(frames[i])

class Line(object):

    def __init__(self, axes, color='#8f1402', alpha=1, lw=2):
        self.axes = axes
        self.line = lines.Line2D([], [], alpha=alpha, color=color, lw=lw)
        self.axes.add_line(self.line)

    def action_drawing(self, i, nframes, fargs=None):
        # if i == 0:
        #     for l in self.axes.lines:
        #         if l.get_label() == 'pdf':
        #             fargs['y2'] = l.get_data()[1]
        xrange = np.linspace(fargs['x1'], fargs['x2'], nframes)
        yrange = np.linspace(fargs['y1'], fargs['y2'], nframes)
        self.line.set_data(xrange[:i+1], yrange[:i+1])

    def action_disap(self, i, nframes, fargs=None):
        if i == 0:
            fargs['alpha'] = self.line.get_alpha()
        frames = np.linspace(fargs['alpha'], 0.5, nframes)
        self.line.set_alpha(frames[i])

    def action_move(self, i, nframes, fargs):
        if i == 0:
            x, y = self.line.get_data()
            fargs['from'] = x[0]
        delta = (fargs["to"] - fargs['from']) / nframes
        x, y = self.line.get_data()
        x += delta
        self.line.set_data(x, y)

class Window(object):

    def __init__(self, axes, left, right, pdf, text):
        self.axes = axes
        self.left = left
        self.right = right
        self.pdf = pdf
        self.text = text

    def fill_between(self, x0, x1):
        x, y = self.pdf.line.get_data()
        idx = np.where((x >= x0) & (x <= x1))[0]
        xlim = self.axes.get_xlim()
        self.axes.collections = []
        self.axes.fill_between(x[idx], y[idx], facecolor="#98b2d3", edgecolor='#013a4b')
        # Restore xlim, because it gets changed somehow
        self.axes.set_xlim(xlim, auto=True)

    def action_change_window(self, i, nframes, fargs):
        if i == 0:
            fargs['from_left'] = self.left
            fargs['from_right'] = self.right

        frames_left = np.linspace(fargs["from_left"], fargs["to_left"], nframes)
        frames_right = np.linspace(fargs["from_right"], fargs["to_right"], nframes)

        if fargs['change_text']:
            proba = self.pdf.get_prob_for_range(frames_left[i],frames_right[i])
            proba = str(int(proba*100.)) + "%"
            self.text.text_obj.set_text(proba)
        self.fill_between(frames_left[i], frames_right[i])
        self.left = frames_left[i]
        self.right = frames_right[i]


class Text(object):

    def __init__(self, axes, text, x, y):
        self.axes = axes
        self.text = text
        self.x = x
        self.y = y
        self.bbox = {'boxstyle' : 'square', 'facecolor' : "#98b2d3",
                    'edgecolor' : "None", 'alpha' : 0}
        self.text_obj = text_obj.Text(x=x, y=y, text=text, alpha=0., color='#013a4b', fontsize=30,
                                      verticalalignment='center', horizontalalignment='center',
                                      bbox=self.bbox)
        self.axes.add_artist(self.text_obj)

    def action_appearing(self, i, nframes, fargs):
        frames = np.linspace(0, fargs['to_alpha'], nframes)
        self.text_obj.set_alpha(frames[i])
        self.bbox['alpha'] = frames[i]
        self.text_obj.set_alpha(frames[i])
        self.text_obj.set_bbox(self.bbox)





