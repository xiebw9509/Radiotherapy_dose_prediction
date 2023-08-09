
#coding:utf8
import visdom
import time
import numpy as np


class Visualizer(object):
    '''
    visdom
    '''
    def __init__(self, env='default', **kwargs):
        '''
        function
        :param env: visdom name
        :param kwargs: other params
        '''
        self.vis = visdom.Visdom(env=env, **kwargs)

        self.names = []
        self.index= {}
        self.log_text = ''

    def reinit(self, env='default', **kwargs):
        '''
        initial
        :param env: visdom name
        :param kwargs: other params
        :return: self
        '''
        self.vis = visdom.Visdom(env=env, **kwargs)
        return self

    def plot_with_x(self, name, x, y, **kwargs):
        '''
        Displays the specified coordinate point in the specified window

        :param name: windows name
        :param x: coordinate x
        :param y: coordinate y
        :param kwargs: other params

        '''
        if name not in self.names:
            self.names.append(name)

        self.vis.line(Y=np.array([y]), X=np.array([x]),
        win=(name),
        opts=dict(title=name),
        update=None if x == 1 else 'append',
        **kwargs
        )

    def plot(self, name, y, **kwargs):
        x = self.index.get(name, 0)
        self.vis.line(Y=np.array([y]), X=np.array([x]),
        win=(name),
        opts=dict(title=name),
        update=None if x == 0 else 'append',
        **kwargs
        )
        self.index[name] = x + 1

    def log(self, info, win='log_text'):
        '''
        logs
        :param info: logs information
        :param win: logs windows
        '''
        self.log_text += ('[{time}] {info} <br>'.format(time=time.strftime('%m%d_%H%M%S'),
                          info=info))

        self.vis.text(self.log_text, win)

    def __getattr__(self, name):
        return getattr(self.vis, name)

