import os


def main():
    if 'CIRCLE_BUILD_NUM' in os.environ:
        return f'_{os.environ["CIRCLE_BUILD_NUM"]}'
    else:
        return '_local_build'


if __name__ == '__main__':
    print(main())
