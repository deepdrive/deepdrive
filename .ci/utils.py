import os

def get_tag_build_id():
    if 'BUILD_ID' in os.environ:
        return f'_{os.environ["BUILD_ID"]}'
    else:
        return '_local_build'
