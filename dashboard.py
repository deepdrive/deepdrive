import json
import sys
import time
from threading import Thread

# TODO: Replace matplotlib UI with Unreal based HUD - and do comms via exsiting Python to Unreal channels

is_py2 = sys.version[0] == '2'
if is_py2:
    import Queue as queue
else:
    import queue as queue
from collections import deque, OrderedDict

import numpy as np
import zmq

import logs

ZMQ_PREFIX = 'deepdrive-dashboard'
ZMQ_CONN_STRING = "tcp://127.0.0.1:5681"

log = logs.get_log(__name__, 'dashboard_log.txt')


def dashboard_fn():
    log.info('Starting dashboard')

    message_q = get_message_q()

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

    def get_next():
        try:
            message = message_q.pop()
            message = message[len(ZMQ_PREFIX) + 1:]
            q_next = OrderedDict(json.loads(message.decode('utf8').replace("'", '"')))
            if q_next.get('should_stop', False):
                print('Stopping dashboard')
                try:
                    anim._fig.canvas._tkcanvas.master.quit()  # Hack to avoid "Exiting Abnormally"
                finally:
                    print('Dashboard stopped')
                    exit()
            else:
                Disp.stats = OrderedDict(q_next['display_stats'])
        except IndexError:
            # Reuuse old stats
            pass
        except KeyboardInterrupt:
            print('KeyboardInterrupt detected in dashboard')

    get_next()
    while not Disp.stats.items():
        get_next()
        time.sleep(0.1)

    font = {'size': 8}
    matplotlib.rc('font', **font)

    log.debug('Populating graph parts with %r', Disp.stats.items())
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
        try:
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
        except KeyboardInterrupt:
            print('KeyboardInterrupt detected in animate, exiting')
            exit()
        return lines

    # TODO: Add blit=True and deal with updating the text if performance becomes unacceptable
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=200, interval=100)

    try:
        plt.show()
    except KeyboardInterrupt:
        print('KeyboardInterrupt detected in show dashboard, exiting')
        exit()


# ZMQ stuff (multiprocessing queue was taking 80ms, so switched to faster zmq) -----------------------------------------

class DashboardPub(object):
    def __init__(self):
        # ZeroMQ Context
        self.context = zmq.Context()

        # Define the socket using the "Context"
        sock = self.context.socket(zmq.PUB)
        sock.bind(ZMQ_CONN_STRING)
        self.sock = sock

    def put(self, display_stats):
        # Message [prefix][message]
        message = "{prefix}-{msg}".format(prefix=ZMQ_PREFIX, msg=json.dumps(list(display_stats.items())))
        log.debug('dashpub put %s', message)
        start = time.time()
        self.sock.send_string(message)
        log.debug('took %fs', time.time() - start)

    def close(self):
        log.info('Closing dashboard')
        self.sock.close()
        self.context.term()


def get_message_q():
    q = deque(maxlen=10)
    zmq_socket = start_zmq_sub()
    thread = Thread(target=poll_zmq, args=(zmq_socket, q))
    thread.start()
    return q


def poll_zmq(zmq_socket, q):
    while True:
        message = zmq_socket.recv()
        log.debug('dashsub pull %s', message)
        q.appendleft(message)


def start_zmq_sub():
    # ZeroMQ Context
    context = zmq.Context()
    # Define the socket using the "Context"
    sock = context.socket(zmq.SUB)
    sock.setsockopt_string(zmq.SUBSCRIBE, ZMQ_PREFIX)
    sock.connect(ZMQ_CONN_STRING)
    return sock
