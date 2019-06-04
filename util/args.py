import argparse


class Args:
    all: argparse.ArgumentParser
    agent: argparse.ArgumentParser

    def __init__(self):
        self.all = argparse.ArgumentParser()
        self.agent = argparse.ArgumentParser()

    def add(self, *args, **kwargs):
        self.all.add_argument(*args, **kwargs)

    def add_agent_arg(self, *args, **kwargs):
        self.agent.add_argument(*args, **kwargs)
        self.all.add_argument(*args, **kwargs)