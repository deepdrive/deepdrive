from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

from future.builtins import (dict, input,
                             str)

import os
import time
import glob
import json

import numpy as np

import config as c
import logs
import utils

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
                 should_record=True, eval_only=False):
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
            mp4_file = utils.hdf5_to_mp4()
            gist_url = utils.upload_to_gist(
                'deepdrive-results-' + c.DATE_STR,
                [c.SUMMARY_CSV_FILENAME, c.EPISODES_CSV_FILENAME])
            self.create_artifacts_inventory(
                gist_url=gist_url,
                hdf5_dir=c.HDF5_SESSION_DIR,
                episodes_file=c.EPISODES_CSV_FILENAME,
                summary_file=c.SUMMARY_CSV_FILENAME,
                mp4_file=mp4_file)

    def save_recordings(self):
        name = str(self.recorded_obz_count // c.FRAMES_PER_HDF5_FILE).zfill(10)
        filepath = os.path.join(self.hdf5_dir, '%s.hdf5' % name)
        utils.save_hdf5(self.obz_recording, filename=filepath, background=True)
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
                                   hdf5_dir: str,
                                   episodes_file: str,
                                   summary_file: str,
                                   mp4_file: str):
        # TODO: Add list of artifacts results file with:
        filename = os.path.join(c.RESULTS_DIR, 'artifact-inventory.json')
        with open(filename, 'w') as out_file:
            observations: list = glob.glob(hdf5_dir + '/*.hdf5')
            data = {'artifacts': {
                'mp4': mp4_file,
                'gist': gist_url,
                'performance_summary': summary_file,
                'episodes': episodes_file,
                'observations': observations,
            }}
            json.dump(data, out_file, indent=2)
        log.info('Wrote artifacts inventory to %s' % filename)
        # TODO: Upload to YouTube on pull request
        # TODO: Save a description file with the episode score summary, gist link, and s3 link

    def num_unsaved_observations(self):
        return self.recorded_obz_count - self.num_saved_observations
