from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import shutil
import sys
from typing import List

from box import Box
from future.builtins import (dict, input,
                             str)

import os
import glob
import json


import config as c
import logs
import utils
from sim.score import TotalScore, EpisodeScore
from util.anonymize import anonymize_user_home

log = logs.get_log(__name__)

TEST_SAVE_IMAGE = False


class Recorder(object):
    """
    Responsible for all artifact creation and uploading, including:
    * HDF5's (Optionally saved to S3)
    * Results csv files (Uploaded to Gist)
    * MP4's (Uploaded to YouTube)
    """

    def __init__(self, recording_dir, should_record_agent_actions=True,
                 should_record=True, eval_only=False, should_upload_gist=False,
                 public=False, main_args=None):
        self.save_threads = []  # type list
        self.record_agent_actions = should_record_agent_actions  # type: bool
        self.should_record = should_record  # type: bool
        self.hdf5_dir = c.HDF5_SESSION_DIR  # type: str
        self.obz_recording = []  # type: list
        self.skipped_first_agent_action = False  # type: bool
        self.was_agent_action = True  # type: bool
        self.recorded_obz_count = 0  # type: int
        self.num_saved_observations = 0  # type: int
        self.recording_dir = recording_dir  # type: str
        self.eval_only = eval_only  # type: bool
        self.should_upload_gist = should_upload_gist  # type: bool
        self.public = public  # type: bool
        self.main_args = main_args  # type: dict
        if self.should_record:
            log.info('Recording driving data to %s', self.hdf5_dir)

    def step(self, obz, is_agent_action=True):
        self.was_agent_action = is_agent_action
        log.debug('obz_exists? %r should_record? %r', obz is not None,
                  self.should_record)
        if self.should_record_obz(obz):
            log.debug('Recording frame')
            self.obz_recording.append(obz)
            if TEST_SAVE_IMAGE:
                utils.save_camera(obz['cameras'][0]['image'],
                                  obz['cameras'][0]['depth'],
                                  self.hdf5_dir, 'screenshot_' +
                                  str(self.step).zfill(10))
                input('continue?')
            self.recorded_obz_count += 1
            if self.recorded_obz_count % 100 == 0:
                log.info('%d recorded observations', self.recorded_obz_count)

        else:
            log.debug('Not recording frame.')
        self.maybe_save()

    def maybe_save(self):
        if (
                self.should_record and
                self.recorded_obz_count != 0 and
                self.recorded_obz_count % c.FRAMES_PER_HDF5_FILE == 0 and
                self.num_saved_observations < self.recorded_obz_count
        ):
            self.save_recordings()

    def close(self, total_score:TotalScore, episode_scores:List[EpisodeScore]):
        log.info('Closing recorder')
        if self.eval_only:
            # Okay to have partial eval recordings
            self.save_unsaved_observations()
        else:
            log.info('Discarding %d observations to keep even number of '
                     'frames in recorded datasets. '
                     'Pass --eval-only to save all observations.' %
                     self.num_saved_observations)

        if self.recorded_obz_count > 0:
            for save_thread in self.save_threads:
                # Wait for HDF5 saves to complete
                save_thread.join()
            mp4_file = utils.hdf5_to_mp4()
            local_public_run = self.should_upload_gist and self.public
            server_public_run = c.UPLOAD_ARTIFACTS
            public_run = local_public_run or server_public_run

            if public_run:
                # Gists will be accessible via YOUTGETMYJIST token
                # regardless of whether they are 'public' gists.
                gist_url = utils.upload_to_gist(
                    'deepdrive-results-' + c.DATE_STR,
                    [c.SUMMARY_CSV_FILENAME, c.EPISODES_CSV_FILENAME],
                    public=True)
                log.info('gist uploaded to %s', gist_url)
            else:
                gist_url = 'not uploaded'

            hdf5_observations = glob.glob(c.HDF5_SESSION_DIR + '/*.hdf5')
            self.create_artifacts_inventory(
                hdf5_observations=hdf5_observations,
                episodes_file=c.EPISODES_CSV_FILENAME,
                summary_file=c.SUMMARY_CSV_FILENAME,
                mp4_file=mp4_file)

            if 'DEEPDRIVE_UPLOAD_ARTIFACTS' in os.environ:
                uploaded = self.upload_artifacts(mp4_file, hdf5_observations)
                youtube_id, mp4_url, hdf5_urls = uploaded
                create_botleague_results(total_score, episode_scores, gist_url,
                                         youtube_id, mp4_url, hdf5_urls,
                                         episodes_file=c.EPISODES_CSV_FILENAME,
                                         summary_file=c.SUMMARY_CSV_FILENAME,)
            # TODO: Create a Botleague compatible results.json file with
            #  - YouTube link
            #  - HDF5 links
            #  - artifacts.json stuff (gist_url) etc..

            # TODO: Add YouTube description file with the episode score summary,
            #  gist link, and s3 link

    def save_recordings(self):
        name = str(self.recorded_obz_count // c.FRAMES_PER_HDF5_FILE).zfill(10)
        filepath = os.path.join(self.hdf5_dir, '%s.hdf5' % name)
        thread = utils.save_hdf5(self.obz_recording, filename=filepath,
                                 background=True)
        self.save_threads.append(thread)
        log.info('Flushing output data')
        self.obz_recording = []
        self.num_saved_observations = self.recorded_obz_count

    def save_unsaved_observations(self):
        if self.should_record and self.num_unsaved_observations():
            self.save_recordings()

    def should_record_obz(self, obz):
        if not obz:
            return False
        elif not self.should_record:
            return False
        elif self.record_agent_actions:
            return self.should_record
        else:
            is_game_driving = self.get_is_game_driving(obz)
            safe_action = is_game_driving and not self.was_agent_action
            if safe_action:
                # TODO: Test to see if skipped_first_agent_action guard
                #  can be safely removed
                if self.skipped_first_agent_action:
                    return True
                else:
                    self.skipped_first_agent_action = True
                    return False
            else:
                self.skipped_first_agent_action = False
                return False

    @staticmethod
    def get_is_game_driving(obz):
        if not obz:
            log.warn(
                'Observation not set, assuming game not driving to '
                'prevent recording bad actions')
            return False
        return obz['is_game_driving'] == 1

    @staticmethod
    def create_artifacts_inventory(hdf5_observations: list,
                                   episodes_file: str,
                                   summary_file: str,
                                   mp4_file: str):
        anon = anonymize_user_home
        p = os.path
        filename = p.join(c.RESULTS_DIR, 'artifacts.json')
        with open(filename, 'w') as out_file:
            data = {'artifacts': {
                'mp4': anon(mp4_file),
                'performance_summary': anon(summary_file),
                'episodes': anon(episodes_file),
                'hdf5_observations': [anon(o) for o in hdf5_observations],
            }}
            json.dump(data, out_file, indent=2)
        log.info('Wrote artifacts inventory to %s' % anon(filename))
        latest_artifacts_filename = p.join(c.RESULTS_BASE_DIR,
                                           'latest-artifacts.json')
        shutil.copy2(filename, latest_artifacts_filename)
        print('\n****\nARTIFACTS INVENTORY COPIED TO: "%s"' +
              anon(latest_artifacts_filename))

    def num_unsaved_observations(self):
        return self.recorded_obz_count - self.num_saved_observations

    @staticmethod
    def upload_artifacts(mp4_file, hdf5_observations) -> (str, str, List[str]):
        youtube_id = utils.upload_to_youtube(mp4_file)
        if youtube_id:
            log.info('Successfully uploaded to YouTube! %s', youtube_id)

        mp4_url = upload_artifacts_to_s3(mp4_file, 'mp4')[0]
        hdf5_urls = upload_artifacts_to_s3(hdf5_observations, 'hdf5')
        return youtube_id, mp4_url, hdf5_urls


def upload_artifacts_to_s3(file_paths, directory) -> List[str]:
    from ue4helpers import AWSUtils
    ret = []
    for file_path in file_paths:
        s3path = 'artifacts/' + os.path.basename(c.RESULTS_DIR)
        key = s3path + ('/%s/' % directory) + os.path.basename(file_path)
        AWSUtils.upload_file(c.AWS_BUCKET, key=key, filename=file_path)
        ret.append('%s%s' % (c.BUCKET_URL, key))
    return ret


def create_botleague_results(total_score, episode_scores, gist_url, youtube_id,
                             mp4_url, hdf5_urls, episodes_file, summary_file,):
    ret = Box()
    ret.score = total_score.median
    ret.youtube_id = youtube_id
    ret.mp4 = mp4_url
    ret.gist = gist_url
    ret.sensorimotor_specific.num_episodes = len(episode_scores)
    ret.problem_specific.summary = summary_file
    ret.problem_specific.episodes = episodes_file
    ret.problem_specific.observations = hdf5_urls
    """
    {

    # Added by botleague liaison
  "problem": "domain_randomization",
  "username": "curie",
  "botname": "forward-agent",
  "status": "success",
  "utc_timestamp": 87600000,
  "docker_digest": "qewrqwerqwer",

  # Added by Adam using docker logs
  "log": "https://gist.githubusercontent.com/crizCraig/38f19d2e3226a822c6ce09ea618ac7ab/raw/32108e3a290a4bccea38b1bade31d6a4b42b32ce/gistfile1.txt",
  
  # Added by Adam
  "docker_image_url": "https://s3-us-west-1.amazonaws.com/ci/build/12341234/agent.tar",
  
  
  "json_commit": "https://github.com/deepdrive/agent-zoo/commit/4a0e6af15c5ee05b62c6705d40aece250112a57d",
  "source_commit": "https://github.com/curie/forward-agent/commit/defc93d95944099d3e61cda6542bb4ffe7a28abf",

  "youtube": "https://www.youtube.com/watch?v=rjZCjosEFpI&t=2575s",
  "mp4": "https://s3-us-west-1.amazonaws.com/ci/build/12341234/asdf.mp4",
  "sensorimotor_specific": {
      "average_fps": 7.9,
      "max_step_milliseconds": 500.134323,
      "total_number_of_steps": 1234,
      "dropped_steps": 34,
      "num_episodes": 1234,
  },

  "driving_specific": {
    "avg_kph": 12.812343,
    "max_g_forces": 4.2,
    "max_kph": 22.33333
  },
  "$comment1": "Non driving tasks are possible but not yet implemented",
  "problem_specific": {
    "summary": "https://s3-us-west-1.amazonaws.com/ci/build/12341234/2019-04-04__11-48-55PM_r0_summary.csv",
    "episodes": "https://s3-us-west-1.amazonaws.com/ci/build/12341234/2019-04-04__11-48-55PM_r0_episodes.csv",
    "observations": [
      "https://s3-us-west-1.amazonaws.com/ci/build/12341234/hdf5/000000.mp4"
    ]
  }
}
    """
    return ret.to_dict()
