from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import shutil
import sys

from future.builtins import (dict, input,
                             str)

import os
import glob
import json


import config as c
import logs
import utils
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

    def close(self):
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
            if self.should_upload_gist and self.public:
                # Gists will be accessible via YOUTGETMYJIST token
                # regardless of whether they are 'public' gists.
                gist_url = utils.upload_to_gist(
                    'deepdrive-results-' + c.DATE_STR,
                    [c.SUMMARY_CSV_FILENAME, c.EPISODES_CSV_FILENAME],
                    public=self.public)
                log.info('gist uploaded to %s', gist_url)
            else:
                gist_url = 'not uploaded'

            hdf5_observations = glob.glob(c.HDF5_SESSION_DIR + '/*.hdf5')
            self.create_artifacts_inventory(
                gist_url=gist_url,
                hdf5_observations=hdf5_observations,
                episodes_file=c.EPISODES_CSV_FILENAME,
                summary_file=c.SUMMARY_CSV_FILENAME,
                mp4_file=mp4_file)

            if 'DEEPDRIVE_UPLOAD_ARTIFACTS' in os.environ:
                self.upload_artifacts(mp4_file, c.HDF5_SESSION_DIR)

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
    def create_artifacts_inventory(gist_url: str,
                                   hdf5_observations: list,
                                   episodes_file: str,
                                   summary_file: str,
                                   mp4_file: str):
        # TODO: Add list of artifacts results file with:
        anon = anonymize_user_home
        p = os.path
        filename = p.join(c.RESULTS_DIR, 'artifacts.json')
        with open(filename, 'w') as out_file:
            data = {'artifacts': {
                'mp4': anon(mp4_file),
                'gist': anon(gist_url),
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

        # TODO: Upload to YouTube on pull request
        # TODO: Save a description file with the episode score summary,
        #  gist link, and s3 link

    def num_unsaved_observations(self):
        return self.recorded_obz_count - self.num_saved_observations

    @staticmethod
    def upload_artifacts(mp4_file, hdf5_observations):
        youtube_id = utils.upload_to_youtube(mp4_file)
        if youtube_id:
            log.info('Successfully uploaded to YouTube! %s', youtube_id)
        from ue4helpers import AWSUtils
        for hdf5_file in hdf5_observations:
            s3path = 'artifacts/' + os.path.basename(c.RESULTS_DIR)
            key = s3path + '/hdf5/' + os.path.basename(hdf5_file)
            AWSUtils.upload_file('deepdrive', key=key, filename=hdf5_file)



