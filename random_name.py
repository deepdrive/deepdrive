import string
from random import choice, randint

ADJECTIVES = [
    'super',
    'novel',
    'deep',
    'recurrent',
    'YOLO',
    'adversarial',
    'evolutional',
    'linguistic',
    'muliagent',
    'auxillary',
    'intrinsic',
    'end-to-end',
    'neural',
    'feed-forward',
    'hierarchical',
    'please-work',
    'this-will-work',
    'hacky',
    'WIP',
    'naive',
    'random',
    'guided',
    'bi-directional',
    'linguistic',
    'Turing',
    'attention',
    'domain',
    'replay-based',
    'unguided',
    'fixed',
    'deterministic',
    'stochastic',
    'large',
    'CNN',
    'dilated',
    'synthetic',
    'recursive',
    'double',
    'real-time',
    'shallow',
    'decision',
    'model-based',
    'model-free',
    'efficient',
    'value',
    'online',
    'offline',
    'memory-based',
    'meta',
    'nonparametric',
    'latent',
    'post-bayesian',
    'Kolmogorov',
    'Solomonoff',
    'WIP',
]

NOUNS = [
    'inception',
    'bugfix',
    'GAN',
    'NN',
    'net',
    'search',
    'LSTM',
    'evolution',
    'optimization',
    'memory',
    'actor-critic',
    'machine',
    'attention',
    'backprop',
    'adaptation',
    'learning',
    'averaging',
    'replay',
    'transfer learning',
    'hack',
    'fix',
    'refactor',
    'test',
    'with control dependencies',
    'tweak'
]


def generate():
    noun = choice(NOUNS)
    if noun in ['NN', 'GAN']:
        prefix_len = randint(1, 1)
        prefix = ''.join([choice(string.ascii_uppercase) for _ in range(prefix_len)])
        noun = prefix + noun
    adjectives = _get_adjectives()
    name = '%s %s' % (adjectives, noun)
    return name


def _get_adjectives():
    rand100 = randint(0, 100)
    if rand100 <= 90:
        num_adjectives = 2
    elif rand100 <= 99:
        num_adjectives = 3
    else:
        num_adjectives = 4
    adjectives = [choice(ADJECTIVES) for _ in range(num_adjectives)]
    if not adjectives[0][0].isupper():
        adjectives[0] = adjectives[0].capitalize()
    return ' '.join(adjectives)


if __name__ == '__main__':
    for i in range(200):
        print(generate())
