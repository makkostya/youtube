import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as path
import matplotlib.lines as lines
import matplotlib.text as text_obj
import matplotlib.collections as colls
from animation import animation as anim
from utils import (set_area, set_pdf_line, set_hist, set_point, set_frame, set_mean_line, set_text,
                   set_interval)
import math
from scipy.stats import norm
import pickle
import io
import time
import gc


def sampling(p1, p2, num, d):
    t = np.linspace(0, 1, num)
    t = t ** d
    sol = np.zeros(num)
    for i in range(num):
        sol[i] = p1 + (p2 - p1) * t[i]
    return sol


class Frame(object):
    def __init__(self, fig=None, axes=None, axes_ratio=1., aspect_ratio=9. / 16., my_dpi=96.):
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
                actions.append(obj.change_scale(t_begin, t_end, ratio))
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

            # delta_after = fargs['x2'] - fargs['x1']
            # delta_before = self.axes.get_xlim()[1] - self.axes.get_xlim()[0]
            # fargs['ratio'] = delta_after / delta_before


        xlim = [fargs['x_min'][i], fargs['x_max'][i]]
        delta_x = xlim[1] - xlim[0]
        y_max = delta_x / self.ax_ratio + fargs['y_min'][i]
        self.axes.set_ylim([fargs['y_min'][i], y_max])
        self.axes.set_xlim(xlim)
        self.axes.set_xticks(range(int(math.ceil(fargs['x_min'][i])), int(math.floor(fargs['x_max'][i]))))

        # for j, obj in enumerate(self.objects):
        #     if isinstance(obj, Point):
        #         fargs['rx'] = obj.rx * fargs['ratio']
        #         obj._change_radius(i, nframes, fargs)

    def ticks_disappear(self, t_begin, t_end):
        return anim.Action([], self._ticks_disappear, t_begin, t_end, {})

    def _ticks_disappear(self, i, nframes, fargs):
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


class DistNorm(object):
    def __init__(self, mean, std, x_range, pdf):
        # Parameters
        self.mean = mean
        self.std = std
        self.x_range = x_range
        self.pdf = pdf
        self.texts = {'left': '', 'center': '', 'right': ''}
        self.left = mean
        self.right = mean
        self.area_status = [0, 1, 0]
        self.y_range = self.pdf(self.x_range, self.mean, self.std)
        self.y_mean = self.pdf(self.mean, self.mean, self.std)

        self.create_artists()

    def create_artists(self):
        self.artists = []

        # Pdf line
        self.pdf_line = lines.Line2D([], [], alpha=set_pdf_line.alpha, color=set_pdf_line.lc, lw=set_pdf_line.lw)
        self.artists = [self.pdf_line]

        # Mean line
        self.mean_line = lines.Line2D([self.mean, self.mean], [0, self.y_mean],
                                      color=set_mean_line.lc, lw=set_mean_line.lw, visible=False)
        self.artists += [self.mean_line]

        # Areas
        self._update_area_verts()
        self.area_collections = []
        for i, k in enumerate(self.area_verts.keys()):
            fc = set_area.fc[1]
            ec = set_area.ec[1]
            if self.area_status:
                fc = set_area.fc[0]
                ec = set_area.ec[0]
            verts = self.area_verts[k]
            collection = colls.PolyCollection([verts], facecolors=fc, edgecolors=ec, visible=False, alpha=1.)
            self.area_collections += [collection]
        self.artists += self.area_collections  # 3 polygons for areas

        # Texts
        p_left = norm.cdf(self.left, self.mean, self.std)
        p_tmp = norm.cdf(self.right, self.mean, self.std)
        p_center = p_tmp - p_left
        p_right = 1 - p_tmp

        self.textboxes = []
        for k in self.texts.keys():
            if k == 'center':
                x = (self.left + self.right) / 2.
                y = 0.9 * (self.area_verts['center'][1, 1] + self.area_verts['center'][-2, 1]) / 2.
                self.texts[k] = str(int(round(100 * p_center))) + '%'
            if k == 'left':
                x = self.left
                y = 1.1 * (self.area_verts['center'][1, 1])
                self.texts[k] = str(int(round(100. * p_left))) + '%'
            if k == 'right':
                x = self.right
                y = 1.1 * (self.area_verts['center'][-2, 1])
                self.texts[k] = str(int(round(100 * p_right))) + '%'
            self.textboxes += [text_obj.Text(x=x, y=y, text=self.texts[k], visible=False,
                                             color=set_text.fc, alpha=1.,
                                             fontsize=set_text.fs, bbox=set_text.bbox,
                                             verticalalignment='center', horizontalalignment='center')]
        self.artists += self.textboxes

        # correct z_orders
        z0 = self.area_collections[0].get_zorder()
        self.mean_line.set_zorder(z0 + 1)
        self.pdf_line.set_zorder(z0 + 2)
        for t in self.textboxes:
            t.set_zorder(z0 + 3)

    def update_artists(self):
        self.y_range = self.pdf(self.x_range, self.mean, self.std)
        self.y_mean = self.pdf(self.mean, self.mean, self.std)

        # Pdf line
        self.pdf_line.set_data(self.x_range, self.y_range)

        # Mean line
        self.mean_line.set_data([self.mean, self.mean], [0, self.y_mean])

        # Areas
        self._update_area_verts()
        for i, k in enumerate(self.area_verts.keys()):
            verts = self.area_verts[k]
            self.area_collections[i].set_verts([verts])

        # Texts
        p_left, p_center, p_right = self._bounds_to_p()

        for i, k in enumerate(self.texts.keys()):
            if k == 'center':
                x = (self.left + self.right) / 2.
                y = 0.7 * self.y_mean
                # y = 0.9 * (self.area_verts['center'][1, 1] + self.area_verts['center'][-2, 1]) / 2.
                self.texts[k] = str(int(round(100 * p_center))) + '%'
            if k == 'left':
                x = self.left
                y = 1.1 * (self.area_verts['center'][1, 1])
                self.texts[k] = str(int(round(100. * p_left))) + '%'
            if k == 'right':
                x = self.right
                y = 1.1 * (self.area_verts['center'][-2, 1])
                self.texts[k] = str(int(round(100 * p_right))) + '%'
            self.textboxes[i].set_position((x,y))
            self.textboxes[i].set_text(self.texts[k])

    def _update_area_verts(self):
        self.area_verts = {}

        # Left
        idx = np.where(self.x_range < self.left)[0]
        verts = np.zeros([len(idx) + 2, 2])
        for j, i in enumerate(idx):
            verts[j + 1, :] = np.array([self.x_range[i], self.y_range[i]])
        verts[0, :] = np.array([self.x_range[idx[0]], 0])
        verts[-1, :] = np.array([self.x_range[idx[-1]], 0])
        self.area_verts['left'] = verts.copy()

        # Center
        idx = np.where((self.x_range >= self.left) & (self.x_range <= self.right))[0]
        if len(idx) == 0:
            idx = [np.where(self.x_range >= self.left)[0][0]]
        verts = np.zeros([len(idx) + 2, 2])
        for j, i in enumerate(idx):
            verts[j + 1, :] = np.array([self.x_range[i], self.y_range[i]])
        verts[0, :] = np.array([self.x_range[idx[0]], 0])
        verts[-1, :] = np.array([self.x_range[idx[-1]], 0])
        self.area_verts['center'] = verts.copy()

        # Right
        idx = np.where(self.x_range > self.right)[0]
        verts = np.zeros([len(idx) + 2, 2])
        for j, i in enumerate(idx):
            verts[j + 1, :] = np.array([self.x_range[i], self.y_range[i]])
        verts[0, :] = np.array([self.x_range[idx[0]], 0])
        verts[-1, :] = np.array([self.x_range[idx[-1]], 0])
        self.area_verts['right'] = verts.copy()

    def _p_to_bounds(self, p):
        p = 1 - p
        self.left = norm.ppf(p / 2, self.mean, self.std)
        self.right = norm.ppf(1 - p / 2, self.mean, self.std)

    def _bounds_to_p(self):
        p_left = norm.cdf(self.left, self.mean, self.std)
        p_tmp = norm.cdf(self.right, self.mean, self.std)
        p_center = p_tmp - p_left
        p_right = 1 - p_tmp
        return [p_left, p_center, p_right]

    def pdf_appear_left_to_right(self, t_begin, t_end, frame):
        return anim.Action([], self._pdf_appear_left_to_right, t_begin, t_end, {'frame': frame})

    def _pdf_appear_left_to_right(self, i, nframes, fargs):
        if i == 0:
            self.pdf_line.set_visible(True)
            range = np.where((self.x_range >= fargs['frame'].axes.get_xlim()[0]) &
                             (self.x_range <= fargs['frame'].axes.get_xlim()[1]))[0]
            fargs['range'] = np.linspace(range[0], range[-1], nframes, dtype='int')

        x = self.x_range[:fargs['range'][i]]
        y = self.pdf(x, self.mean, self.std)
        self.pdf_line.set_data(x, y)

        if i == nframes - 1:
            self.pdf_line.set_data(self.x_range, self.y_range)

    def area_appear_from_mean(self, t_begin, t_end, p):
        return anim.Action([], self._area_appear_from_mean, t_begin, t_end, {'p': p})

    def _area_appear_from_mean(self, i, nframes, fargs):
        if i == 0:
            self._p_to_bounds(fargs['p'])
            fargs['range_left'] = np.linspace(self.mean, self.left, nframes)
            fargs['range_right'] = np.linspace(self.mean, self.right, nframes)
            for j, k in enumerate(self.area_verts.keys()):
                if k == 'center':
                    self.area_collections[j].set_visible(True)
        self.left = fargs['range_left'][i]
        self.right = fargs['range_right'][i]
        self.update_artists()

    def mean_line_appear(self, t_begin, t_end):
        return anim.Action([], self._mean_line_appear, t_begin, t_end, {})

    def _mean_line_appear(self, i, nframes, fargs):
        if i == 0:
            self.mean_line.set_visible(True)
            fargs['frames'] = np.linspace(0, self.y_mean, nframes)
        self.mean_line.set_data([self.mean, self.mean], [0, fargs['frames'][i]])

    def move(self, t_begin, t_end, move_to):
        return anim.Action([], self._move, t_begin, t_end, {'move_to': move_to})

    def _move(self, i, nframes, fargs):
        if i == 0:
            fargs['frames'] = np.linspace(self.mean, fargs['move_to'], nframes)
        delta = fargs['frames'][i] - self.mean
        self.left += delta
        self.right += delta
        self.mean = fargs['frames'][i]
        self.update_artists()

    def mean_line_change_alpha(self, t_begin, t_end, alpha=0):
        return anim.Action([], self._mean_line_change_alpha, t_begin, t_end, {'alpha': alpha})

    def _mean_line_change_alpha(self, i, nframes, fargs):
        if i == 0:
            fargs['frames'] = np.linspace(1., fargs['alpha'], nframes)
        self.mean_line.set_alpha(fargs['frames'][i])

    def text_appear(self, t_begin, t_end, idx=1):
        return anim.Action([], self._text_appear, t_begin, t_end, {'idx': idx})

    def _text_appear(self, i, nframes, fargs):
        if i == 0:
            fargs['frames'] = np.linspace(0, 1, nframes)
            self.textboxes[fargs['idx']].set_visible(True)
        idx = fargs['idx']
        self.textboxes[idx].set_alpha(fargs['frames'][i])
        bbox = set_text.bbox.copy()
        bbox['alpha'] = fargs['frames'][i]
        self.textboxes[idx].set_bbox(bbox)

    def p_change(self, t_begin, t_end, p):
        return anim.Action([], self._p_change, t_begin, t_end, {'p': p})

    def _p_change(self, i, nframes, fargs):
        if i == 0:
            _, p, _ = self._bounds_to_p()
            fargs['frames'] = np.linspace(p, fargs['p'], nframes)
        self._p_to_bounds(fargs['frames'][i])
        self.update_artists()

    def all_disappear(self, t_begin, t_end):
        return anim.Action([], self._all_disappear, t_begin, t_end, {})

    def _all_disappear(self, i, nframes, fargs):
        if i == 0:
            fargs['range'] = np.linspace(1, 0, nframes)
            alphas = []
            for art in self.artists:
                alphas.append(art.get_alpha())
            fargs['alphas'] = alphas

        for j, art in enumerate(self.artists):
            art.set_alpha(fargs['alphas'][j] * fargs['range'][i])
            if i == nframes - 1:
                art.set_visible(False)


class Point(object):
    def __init__(self, x=0, y=0, rx=set_point.rx, aspect=9. / 16., p2c_ratio=6.):
        self.x = x
        self.y = y
        self.rx = rx
        self.aspect = aspect
        self.p2c_ratio = p2c_ratio
        self.scaling_ratio = 1
        self.point = patches.Ellipse((self.x, self.y), self.rx, self.rx / self.aspect, visible=False,
                                     facecolor=set_point.fc, edgecolor=set_point.ec, alpha=set_point.alpha)

        self.center = patches.Ellipse((self.x, self.y), self.rx / p2c_ratio,
                                                    self.rx / self.aspect / p2c_ratio,
                                                    visible=False, facecolor=set_point.center_c)
        self.point.set_zorder(10)
        self.center.set_zorder(11)
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

    def change_scale(self, t_begin, t_end, ratio):
        return anim.Action([], self._change_scale, t_begin, t_end, {'ratio': ratio})

    def _change_scale(self, i, nframes, fargs):
        if i == 0:
            fargs['frames'] = np.linspace(self.rx, self.rx * fargs['ratio'], nframes)
            self.scaling_ratio = fargs['ratio']
        self.update_radius(fargs['frames'][i])

    def appear(self, t_begin, t_end, rx=set_point.rx, ratio=0.7, scale=1.3):
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
            frames_1 = np.linspace(0, fargs['rx'] * scale * self.scaling_ratio, nframes_1)
            frames_2 = np.linspace(fargs['rx'] * scale * self.scaling_ratio, fargs['rx'] * self.scaling_ratio,
                                   nframes_2)
            frames = np.hstack([frames_1, frames_2])
            fargs['frames'] = frames
        self.update_radius(fargs['frames'][i])

    def disappear(self, t_begin, t_end):
        return anim.Action([], self._disappear, t_begin, t_end, {})

    def _disappear(self, i, nframes, fargs):
        if i == 0:
            fargs['frames'] = np.linspace(self.rx, 0, nframes)
        self.update_radius(fargs['frames'][i])
        if i == nframes - 1:
            self.point.set_visible(False)
            self.center.set_visible(False)


class Interval(object):
    def __init__(self, left, right, height):
        self.left = left
        self.right = right
        self.height = height
        self.update_verts(self.height)
        self.create_artists()

    def update_verts(self, height):
        self.verts = np.zeros([4, 2])
        self.verts[0, :] = np.array([self.left, -height])
        self.verts[1, :] = np.array([self.left, height])
        self.verts[2, :] = np.array([self.right, height])
        self.verts[3, :] = np.array([self.right, -height])
        self.verts[:, 1] *= set_interval.scale

    def create_artists(self):
        self.line_left = lines.Line2D([], [], color=set_interval.lc, lw=set_interval.lw, visible=False)
        self.line_right = lines.Line2D([], [], color=set_interval.lc, lw=set_interval.lw, visible=False)
        self.collection = colls.PolyCollection([self.verts], facecolors=set_interval.fc, edgecolors=set_interval.ec,
                                               visible=False, alpha=set_interval.alpha)

        self.line_left.set_zorder(10)
        self.line_right.set_zorder(10)
        self.collection.set_zorder(2)

        self.artists = [self.line_left, self.line_right, self.collection]

    def update_artists(self):
        self.collection.set_verts([self.verts])
        self.line_right.set_data([self.right, self.right], [self.height, self.height])
        self.line_left.set_data([self.left, self.left], [self.height, self.height])

    def line_left_appear(self, t_begin, t_end):
        return anim.Action([], self._line_left_appear, t_begin, t_end, {})

    def _line_left_appear(self, i, nframes, fargs):
        if i == 0:
            fargs['range'] = np.linspace(0, self.height, nframes)
            self.line_left.set_visible(True)
        self.line_left.set_data([self.left, self.left], [fargs['range'][i], -fargs['range'][i]])

    def line_right_appear(self, t_begin, t_end):
        return anim.Action([], self._line_right_appear, t_begin, t_end, {})

    def _line_right_appear(self, i, nframes, fargs):
        if i == 0:
            fargs['range'] = np.linspace(0, self.height, nframes)
            self.line_right.set_visible(True)
        self.line_right.set_data([self.right, self.right], [fargs['range'][i], -fargs['range'][i]])

    def area_appear(self, t_begin, t_end):
        return anim.Action([], self._area_appear, t_begin, t_end, {})

    def _area_appear(self, i, nframes, fargs):
        if i == 0:
            fargs['range'] = np.linspace(0, self.height, nframes)
            self.collection.set_visible(True)
        self.update_verts(fargs['range'][i])
        self.collection.set_verts([self.verts])





