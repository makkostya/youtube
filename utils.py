class Plot_settings(object):
    def __init__(self):
        pass
# Frame settings
set_frame = Plot_settings()
set_frame.fc = '#f7f8f2'
set_frame.ticks_ls = 22
set_frame.ticks_lc = '#013a4b'


# Area settings
set_area = Plot_settings()
set_area.fc = ["#98b2d3", 'red']
set_area.ec = ["#013a4b", 'darkred']
set_area.lw = 1
set_area.alpha = 1.

# pdf line settings
set_pdf_line = Plot_settings()
set_pdf_line.lc = '#e55645'
set_pdf_line.lw = 5
set_pdf_line.alpha = 1.

# Hist settings
set_hist = Plot_settings()
set_hist.fc = "#98b2d3"
set_hist.ec = "#013a4b"
set_hist.lw = 1
set_hist.alpha = 1.

# Point settings
set_point = Plot_settings()
set_point.fc = 'darkred'
set_point.ec = 'None'
set_point.center_c = '#013a4b'
set_point.alpha = 0.7
set_point.rx = 0.3

# Mean line settings
set_mean_line = Plot_settings()
set_mean_line.lw = 2
set_mean_line.lc = '#013a4b'

# Text
set_text = Plot_settings()
set_text.fs = 30
set_text.fc = '#013a4b'
set_text.bbox = {'boxstyle' : 'square', 'facecolor' : "#98b2d3",
                'edgecolor' : "None", 'alpha' : 0}

# Interval settings
set_interval = Plot_settings()
set_interval.fc = "#98b2d3"
set_interval.lc = '#013a4b'
set_interval.lw = 3
set_interval.ec = 'None'
set_interval.alpha = 0.7
set_interval.scale = 0.5