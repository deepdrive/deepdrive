import queue
from collections import deque, OrderedDict

import numpy as np

import logs


log = logs.get_log(__name__)

def dashboard_fn(dash_queue):
    print('DEBUG - starting dashboard')
    import matplotlib.animation as animation
    import matplotlib
    try:
        # noinspection PyUnresolvedReferences
        import matplotlib.pyplot as plt
    except ImportError as e:
        log.error('\n\n\n***** Error: Could not start dashboard: %s\n\n', e)
        return
    plt.figure(0)

    class Disp(object):
        stats = {}
        txt_values = {}
        lines = {}
        x_lists = {}
        y_lists = {}

    def get_next(block=False):
        try:
            q_next = dash_queue.get(block=block)
            if q_next['should_stop']:
                print('Stopping dashboard')
                try:
                    anim._fig.canvas._tkcanvas.master.quit()  # Hack to avoid "Exiting Abnormally"
                finally:
                    exit()
            else:
                Disp.stats = q_next['display_stats']
        except queue.Empty:
            # Reuuse old stats
            pass

    get_next(block=True)

    font = {'size': 8}

    matplotlib.rc('font', **font)

    for i, (stat_name, stat) in enumerate(Disp.stats.items()):
        stat = Disp.stats[stat_name]
        stat_label_subplot = plt.subplot2grid((len(Disp.stats), 3), (i, 0))
        stat_value_subplot = plt.subplot2grid((len(Disp.stats), 3), (i, 1))
        stat_graph_subplot = plt.subplot2grid((len(Disp.stats), 3), (i, 2))
        stat_label_subplot.text(0.5, 0.5, stat_name, fontsize=12, va="center", ha="center")
        txt_value = stat_value_subplot.text(0.5, 0.5, '', fontsize=12, va="center", ha="center")
        Disp.txt_values[stat_name] = txt_value
        stat_graph_subplot.set_xlim([0, 200])
        stat_graph_subplot.set_ylim([stat['ymin'], stat['ymax']])
        Disp.lines[stat_name], = stat_graph_subplot.plot([], [])
        stat_label_subplot.axis('off')
        stat_value_subplot.axis('off')
        Disp.x_lists[stat_name] = deque(np.linspace(200, 0, num=400))
        Disp.y_lists[stat_name] = deque([-1] * 400)
        frame1 = plt.gca()
        frame1.axes.get_xaxis().set_visible(False)

    plt.subplots_adjust(hspace=0.88)
    fig = plt.gcf()
    fig.set_size_inches(5.5, len(Disp.stats) * 0.5)
    fig.canvas.set_window_title('Dashboard')
    anim = None

    def init():
        lines = []
        for s_name in Disp.stats:
            line = Disp.lines[s_name]
            line.set_data([], [])
            lines.append(line)
        return lines

    def animate(_i):
        lines = []
        get_next()
        for s_name in Disp.stats:
            s = Disp.stats[s_name]
            xs = Disp.x_lists[s_name]
            ys = Disp.y_lists[s_name]
            tv = Disp.txt_values[s_name]
            line = Disp.lines[s_name]
            val = s['value']
            total = s['total']
            tv.set_text(str(round(total, 2)) + s['units'])
            ys.pop()
            ys.appendleft(val)
            line.set_data(xs, ys)
            lines.append(line)
        plt.draw()
        return lines

    # TODO: Add blit=True and deal with updating the text if performance becomes unacceptable
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=200, interval=100)
    plt.show()
