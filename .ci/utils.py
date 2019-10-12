import os

def get_tag_build_id():
    if 'CIRCLE_BUILD_NUM' in os.environ:
        return f'_{os.environ["CIRCLE_BUILD_NUM"]}'
    else:
        return '_local_build'
