import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as path
import matplotlib.lines as lines
import matplotlib.text as text_obj
import matplotlib.collections as colls
from animation import animation as anim
from utils import (set_area, set_pdf_line, set_hist, set_point, set_frame, set_mean_line, set_text)
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

class Frame(object):
    def __init__(self, fig=None, axes=None, axes_ratio=1., aspect_ratio=9./16., my_dpi=96.):
        if fig is None:
            fig, axes = plt.subplots(facecolor='black', figsize=(1920 / my_dpi, 1080 / my_dpi), dpi=my_dpi)
        self.fig = fig
        self.axes = axes
        self.ax_ratio = axes_ratio
        self.aspect_ratio = aspect_ratio
        self.axes.set_aspect(self.ax_ratio * self.aspect_ratio)
        self.axes.set_facecolor(set_frame.fc)
        self.axes.tick_params(top='off', bottom='on', left='off', right='off',
                                labelleft='off', labelbottom='on')
        for i in range(4):
            if i != 2:
                list(self.axes.spines.values())[i].set_visible(False)
        self.axes.spines['bottom'].set_position('zero')
        self.axes.xaxis.set_ticks_position('bottom')
        self.axes.tick_params(axis='x', labelsize=set_frame.ticks_ls, labelcolor=set_frame.ticks_lc)
        self.axes.set_axisbelow(True)

        self.objects = []

    def update_axes_ratio(self):
        delta_y = (self.axes.get_ylim()[1] - self.axes.get_ylim()[0])
        delta_x = (self.axes.get_xlim()[1] - self.axes.get_xlim()[0])
        self.ax_ratio = delta_x / delta_y
        self.axes.set_aspect(self.ax_ratio * self.aspect_ratio)

    def add_objects(self, objs):
        for obj in objs:
            if isinstance(obj, Point):
                obj.change_aspect(self.ax_ratio * self.aspect_ratio)
            self.objects.append(obj)
            for art in obj.artists:
                self.axes.add_artist(art)

    def zoom(self, t_begin, t_end, x1, x2, y1=None):
        actions = [anim.Action([], self._zoom, t_begin, t_end, {'x1': x1, 'x2': x2, 'y1': y1})]
        delta_after = x2 - x1
        delta_before = self.axes.get_xlim()[1] - self.axes.get_xlim()[0]
        ratio = delta_after / delta_before
        for j, obj in enumerate(self.objects):
            if isinstance(obj, Point):
                actions.append(obj.change_radius(t_begin, t_end, obj.rx * ratio))
        return actions

    def _zoom(self, i, nframes, fargs):
        if i == 0:
            xlim = self.axes.get_xlim()
            ylim = self.axes.get_ylim()
            fargs['x_min'] = np.linspace(xlim[0], fargs['x1'], nframes)
            fargs['x_max'] = np.linspace(xlim[1], fargs['x2'], nframes)
            if fargs['y1'] is None:
                fargs['y1'] = ylim[0]
            fargs['y_min'] = np.linspace(ylim[0], fargs['y1'], nframes)

        xlim = [fargs['x_min'][i], fargs['x_max'][i]]
        delta_x = xlim[1] - xlim[0]
        y_max = delta_x / self.ax_ratio + fargs['y_min'][i]
        self.axes.set_ylim([fargs['y_min'][i], y_max])
        self.axes.set_xlim(xlim)
        self.axes.set_xticks(range(int(math.ceil(fargs['x_min'][i])), int(math.floor(fargs['x_max'][i]))))

    def ticks_disappear(self, t_begin, t_end):
        return anim.Action([], self._ticks_disappear, t_begin, t_end, {})

    def _ticks_disappear(self,  i, nframes, fargs):
        if i == 0:
            fargs['alphas'] = np.linspace(1., 0, nframes)
        for l in self.axes.get_xticklabels(which='both'):
            l.set_alpha(fargs['alphas'][i])

class Hist(object):
    def __init__(self, bins, values, data):
        self.bins = bins.copy()
        self.delta = (bins.max() - bins.min()) / len(values)
        self.scale = self.delta * np.sum(values)
        self.values = values / self.scale
        self.data = data.copy()
        self.create_artists()

    def create_artists(self):
        left = np.array(self.bins[:-1]) + 0.1 * (self.bins[1] - self.bins[0])
        right = np.array(self.bins[1:]) - 0.1 * (self.bins[1] - self.bins[0])
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
        self.patch = patches.PathPatch(barpath, facecolor=set_hist.fc, visible=False,
                                       edgecolor=set_hist.ec, linewidth=set_hist.lw, alpha=set_hist.alpha)
        self.artists = [self.patch]

    def appear_down_to_up(self, t_begin, t_end, accel=1.):
        return anim.Action([], self._appear_down_to_up, t_begin, t_end, {'accel': accel})

    def _appear_down_to_up(self, i, nframes, fargs):
        if i == 0:
            self.patch.set_visible(True)
            fargs['frames'] = sampling(0, len(self.data), nframes, fargs['accel'])
        values, bins = np.histogram(self.data[:int(fargs['frames'][i]) + 1], bins=self.bins)
        self.values = values / self.scale
        self.verts[1::5, 1] = self.values
        self.verts[2::5, 1] = self.values

    def disappear_up_to_down(self, t_begin, t_end, accel=1.):
        return anim.Action([], self._disappear_up_to_down, t_begin, t_end, {'accel': accel})

    def _disappear_up_to_down(self, i, nframes, fargs):
        if i == 0:
            self.patch.set_visible(True)
            fargs['frames'] = sampling(len(self.data), 0, nframes, fargs['accel'])
        values, bins = np.histogram(self.data[:int(fargs['frames'][i]) + 1], bins=self.bins)
        self.values = values / self.scale
        self.verts[1::5, 1] = self.values
        self.verts[2::5, 1] = self.values

    def disappear_trans(self, t_begin, t_end):
        return anim.Action([], self._disappear_trans, t_begin, t_end)

    def _disappear_trans(self, i, nframes, fargs):
        if i == 0:
            self.patch.set_visible(True)
            fargs['alphas'] = np.linspace(1, 0, nframes)
        self.patch.set_alpha(fargs['alphas'][i])


class Pdf(object):
    def __init__(self, mean, std, x_range, pdf, alpha=None):
        self.mean = mean
        self.std = std
        self.x_range = x_range
        self.pdf = pdf
        self.alpha = alpha
        self.texts = ['', '', '']
        if alpha is None:
            self.y_range = self.pdf(x_range, mean, std)
            self.y_mean = self.pdf(mean, mean, std)

        self.pdf_line = lines.Line2D([], [], alpha=set_pdf_line.alpha, color=set_pdf_line.lc, lw=set_pdf_line.lw)
        self.area = Area(self.mean, self.mean, self.x_range, self.y_range)
        # self.interval = Interval()
        self.mean_line = Mean_line(self.mean, self.y_mean)

        # text boxes
        for i in range(len(self.texts)):
            k = self.area.verts.keys()[i]
            if k == 'center':
                x = (self.area.left + self.area.right) / 2.
                y = 0.9 * (self.area.verts['center'][1, 1] + self.area.verts['center'][-2, 1]) / 2.
            if k == 'left':
                x = self.area.left
                y = 1.1 * (self.area.verts['center'][1, 1])
            if k == 'right':
                x = self.area.right
                y = 1.1 * (self.area.verts['center'][-2, 1])
            self.textboxes += [text_obj.Text(x=x, y=y, text=self.texts[i], visible=False,
                                            color=set_text.fc,
                                            fontsize=set_text.fs, bbox=set_text.bbox,
                                            verticalalignment='center', horizontalalignment='center')]
        self.create_artists()

    def create_artists(self):
        self.artists = [self.pdf_line] # pdf line
        self.artists += self.area.artists # 3 polygons for areas
        self.artists += [self.mean_line.line]
        self.artists += self.textboxes

    def appear_left_to_right(self, t_begin, t_end, frame):
        return anim.Action([], self._appear_left_to_right, t_begin, t_end, {'frame': frame})

    def _appear_left_to_right(self, i, nframes, fargs):
        if i == 0:
            self.pdf_line.set_visible(True)
            range = np.where((self.x_range >= fargs['frame'].axes.get_xlim()[0]) &
                              (self.x_range <= fargs['frame'].axes.get_xlim()[1]))[0]
            fargs['range'] = np.linspace(range[0], range[-1], nframes,dtype='int')

        x = self.x_range[:fargs['range'][i]]
        if self.alpha is None:
            y = self.pdf(x, self.mean, self.std)
        self.pdf_line.set_data(x, y)

        if i == nframes - 1:
            self.pdf_line.set_data(self.x_range, self.y_range)

    def appear_area_from_mean(self, t_begin, t_end, p):
        return anim.Action([], self._appear_area_from_mean, t_begin, t_end, {'p': p})

    def _appear_area_from_mean(self, i, nframes, fargs):
        if i == 0:
            p = fargs['p']
            left = norm.ppf(p / 2, self.mean, self.std)
            right = norm.ppf(1 - p / 2, self.mean, self.std)
            fargs['range_left'] = np.linspace(self.mean, left, nframes)
            fargs['range_right'] = np.linspace(self.mean, right, nframes)
            for j, k in enumerate(self.area.verts.keys()):
                if k == 'center':
                    self.area.artists[j].set_visible(True)
        self.area.left = fargs['range_left'][i]
        self.area.right = fargs['range_right'][i]
        self.area.update_artists()

    def move(self, t_begin, t_end, move_to):
        return anim.Action([], self._move, t_begin, t_end, {'move_to': move_to})

    def _move(self, i, nframes, fargs):
        if i == 0:
            fargs['frames'] = np.linspace(self.mean, fargs['move_to'], nframes)
        delta = fargs['frames'][i] - self.mean
        self.mean = fargs['frames'][i]
        self.y_range = self.pdf(self.x_range, self.mean, self.std)
        self.y_mean = self.pdf(self.mean, self.mean, self.std)

        # update area
        self.area.y = self.y_range
        self.area.x = self.x_range
        self.area.left += delta
        self.area.right += delta
        self.area.update_artists()


        # update pdf line
        self.pdf_line.set_data(self.x_range, self.y_range)

        # update mean line
        self.mean_line.x = self.mean
        self.mean_line.y = self.y_mean
        self.mean_line.update_artists()

    def appear_mean_line(self, t_begin, t_end):
        return anim.Action([], self._appear_mean_line, t_begin, t_end, {})

    def _appear_mean_line(self, i, nframes, fargs):
        if i == 0:
            self.mean_line.line.set_visible(True)
            fargs['frames'] = np.linspace(0, self.y_mean, nframes)
        self.mean_line.y = fargs['frames'][i]
        self.mean_line.update_artists()

    def change_alpha_mean_line(self, t_begin, t_end, alpha=0):
        return anim.Action([], self._change_alpha_mean_line, t_begin, t_end, {'alpha': alpha})

    def _change_alpha_mean_line(self, i, nframes, fargs):
        if i == 0:
            fargs['frames'] = np.linspace(1., fargs['alpha'], nframes)
        self.mean_line.line.set_alpha(fargs['frames'][i])

    def appear_text(self, t_begin, t_end, idx=1):
        return anim.Action([], self._appear_text, t_begin, t_end, {'idx': idx})

    def _appear_text(self,  i, nframes, fargs):
        if i == 0:
            fargs['frames'] = np.linspace(0, 1, nframes)
            self.textboxes[fargs['idx']].set_visible(True)
        idx = fargs['idx']
        self.textboxes[idx].set_alpha(fargs['frames'][i])
        bbox = set_text.bbox.copy()
        bbox['alpha'] = fargs['frames'][i]
        self.textboxes[idx].set_bbox(bbox)


class Area(object):
    def __init__(self, left=0, right=0, x=[], y=[]):
        self.left = left
        self.right = right
        self.x = x
        self.y = y
        self.area_status = [0, 1, 0]
        self.textboxes = []
        self.create_artists()

    def _update_verts(self):
        self.verts = {}

        # Left
        idx = np.where(self.x < self.left)[0]
        verts = np.zeros([len(idx) + 2, 2])
        for j, i in enumerate(idx):
            verts[j + 1, :] = np.array([self.x[i], self.y[i]])
        verts[0, :] = np.array([self.x[idx[0]], 0])
        verts[-1, :] = np.array([self.x[idx[-1]], 0])
        self.verts['left'] = verts.copy()

        # Center
        idx = np.where((self.x >= self.left) & (self.x <= self.right))[0]
        if len(idx) == 0:
            idx = [np.where(self.x >= self.left)[0][0]]
        verts = np.zeros([len(idx) + 2, 2])
        for j, i in enumerate(idx):
            verts[j+1,:] = np.array([self.x[i], self.y[i]])
        verts[0, :] = np.array([self.x[idx[0]], 0])
        verts[-1, :] = np.array([self.x[idx[-1]], 0])
        self.verts['center'] = verts.copy()

        # Right
        idx = np.where(self.x > self.right)[0]
        verts = np.zeros([len(idx) + 2, 2])
        for j, i in enumerate(idx):
            verts[j + 1, :] = np.array([self.x[i], self.y[i]])
        verts[0, :] = np.array([self.x[idx[0]], 0])
        verts[-1, :] = np.array([self.x[idx[-1]], 0])
        self.verts['right'] = verts.copy()

    def create_artists(self):
        self._update_verts()
        self.artists = []
        self.collections = []
        for i, k in enumerate(self.verts.keys()):
            fc = set_area.fc[1]
            ec = set_area.ec[1]
            if self.area_status:
                fc = set_area.fc[0]
                ec = set_area.ec[0]
            verts = self.verts[k]
            collection = colls.PolyCollection([verts], facecolors=fc, edgecolors=ec, visible=False)
            self.artists += [collection]
            self.collections += [collection]


    def update_artists(self):
        self._update_verts()
        for i, k in enumerate(self.verts.keys()):
            verts = self.verts[k]
            self.collections[i].set_verts([verts])
            if k == 'center':
                x = (self.left + self.right) / 2.
                y = 0.9 * (self.verts['center'][1, 1] + self.verts['center'][-2, 1]) / 2.
            if k == 'left':
                x = self.left
                y = 1.1 * (self.verts['center'][1, 1])
            if k == 'right':
                x = self.right
                y = 1.1 * (self.verts['center'][-2, 1])
            self.textboxes[i].set_position((x,y))
            self.textboxes[i].set_text(self.texts[i])


class Interval(object):
    def __init__(self):
        self.x = 0


class Mean_line(object):
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y
        self.line = lines.Line2D([], [], color=set_mean_line.lc, lw=set_mean_line.lw, visible=False)
        self.create_artists()

    def create_artists(self):
        self.artists = [self.line]

    def appear_down_to_up(self, t_begin, t_end, y=0):
        return anim.Action([], self._appear_down_to_up, t_begin, t_end, {'y' : y})

    def _appear_down_to_up(self, i, nframes, fargs):
        if i == 0:
            self.line.set_visible(True)
            fargs['frames'] = np.linspace(0, fargs['y'], nframes)
        self.y = fargs['frames'][i]
        self.update_artists()

    def update_artists(self):
        self.line.set_data([self.x, self.x], [0, self.y])

class Prob_text(object):
    def __init__(self, text='', x=0, y=0):
        self.x = x
        self.y = y
        self.text = text
        self.artists = self.create_artists()


class Point(object):
    def __init__(self, x=0, y=0, rx=0, aspect=9./16., p2c_ratio = 6.):
        self.x = x
        self.y = y
        self.rx = rx
        self.aspect = aspect
        self.p2c_ratio = p2c_ratio
        self.point = patches.Ellipse((self.x, self.y), self.rx, self.rx / self.aspect, visible=False,
                                     facecolor=set_point.fc, edgecolor=set_point.ec, alpha=set_point.alpha)

        self.center = self.center = patches.Ellipse((self.x, self.y), self.rx / p2c_ratio,
                                                    self.rx / self.aspect / p2c_ratio,
                                                    visible=False, facecolor=set_point.center_c)
        self.create_artists()

    def create_artists(self):
        self.artists = [self.point, self.center]

    def change_aspect(self, aspect):

        self.point.height = self.rx / aspect
        self.center.height = self.point.height / self.p2c_ratio
        self.aspect = aspect

    def update_radius(self, rx):
        self.rx = rx
        self.point.width = self.rx
        self.point.height = self.rx / self.aspect
        self.center.width = self.point.width / self.p2c_ratio
        self.center.height = self.point.height / self.p2c_ratio

    def change_radius(self, t_begin, t_end, rx):
        return anim.Action([], self._change_radius, t_begin, t_end, {'rx': rx})

    def _change_radius(self, i, nframes, fargs):
        if i == 0:
            fargs['frames'] = np.linspace(self.rx, fargs['rx'], nframes)
        self.update_radius(fargs['frames'][i])

    def appear(self, t_begin, t_end, rx=None, ratio=0.7, scale=1.3):
        if rx is None:
            rx = self.rx
        return anim.Action([], self._appear, t_begin, t_end, {'ratio': ratio, 'scale': scale, 'rx': rx})

    def _appear(self, i, nframes, fargs):
        ratio = fargs['ratio']
        scale = fargs['scale']
        if i == 0:
            self.rx = 0
            self.point.set_visible(True)
            self.center.set_visible(True)
            nframes_1 = int(ratio * nframes)
            nframes_2 = nframes - nframes_1
            frames_1 = np.linspace(0, fargs['rx'] * scale, nframes_1)
            frames_2 = np.linspace(fargs['rx'] * scale, fargs['rx'], nframes_2)
            frames = np.hstack([frames_1, frames_2])
            fargs['frames'] = frames
        self.update_radius(fargs['frames'][i])
        # self.rx = fargs['frames'][i]
        # self.point.width = self.rx
        # self.point.height = self.rx / self.aspect
        # self.center.width = self.point.width / self.p2c_ratio
        # self.center.height = self.point.height / self.p2c_ratio

    def disappear(self, t_begin, t_end):
        return anim.Action([], self._disappear, t_begin, t_end, {})

    def _disappear(self, i, nframes, fargs):
        if i == 0:
            fargs['frames'] = np.linspace(self.rx, 0, nframes)
        self.update_radius(fargs['frames'][i])
        if i == nframes - 1:
            self.point.set_visible(False)
            self.center.set_visible(False)

        # self.rx = fargs['frames'][i]
        # self.point.width = self.rx
        # self.point.height = self.rx / self.aspect
        # self.center.width = self.point.width / self.p2c_ratio
        # self.center.height = self.point.height / self.p2c_ratio




