import shutil
from typing import List, Tuple

import requests
from box import Box


import os
import glob
import json
from os.path import join, basename

import config as c
import logs
import utils
from sim.score import TotalScore, EpisodeScore
from util.anonymize import anonymize_user_home
from utils import copy_dir_clean

log = logs.get_log(__name__)

TEST_SAVE_IMAGE = False


class Recorder(object):
    """
    Responsible for artifact creation and uploading, including:
    * HDF5's (Optionally saved to S3)
    * Results csv files (Uploaded to Gist)
    * MP4's (Uploaded to YouTube)
    """

    def __init__(self, recording_dir, should_record_agent_actions=True,
                 should_record=True, eval_only=False, should_upload_gist=False,
                 public=False, main_args=None, is_botleague=False):
        self.save_threads:list = []
        self.record_agent_actions:bool = should_record_agent_actions
        self.should_record:bool = should_record
        self.hdf5_dir:str = c.HDF5_SESSION_DIR
        self.obz_recording:list = []
        self.skipped_first_agent_action:bool = False
        self.was_agent_action:bool = True
        self.recorded_obz_count:int = 0
        self.num_saved_observations:int = 0
        self.recording_dir:str = recording_dir
        self.eval_only:bool = eval_only
        self.should_upload_gist:bool = should_upload_gist
        self.public:bool = public
        self.main_args:dict = main_args
        self.is_botleague:bool = is_botleague
        if self.should_record:
            log.info('Recording driving data to %s', self.hdf5_dir)

    def step(self, obz, done, reward, action, is_agent_action=True):
        self.was_agent_action = is_agent_action
        log.debug('obz_exists? %r should_record? %r', obz is not None,
                  self.should_record)
        if self.should_record_obz(obz):
            log.debug('Recording frame')
            obz['gym_done'] = done
            obz['gym_reward'] = reward
            obz['gym_action'] = action.serialize()
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

    def close(self, total_score:TotalScore, episode_scores:List[EpisodeScore],
              median_fps:float, ):
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
            server_public_run = self.is_botleague
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

            if self.is_botleague:
                create_botleague_results(total_score, episode_scores, gist_url,
                                         hdf5_observations,
                                         mp4_file,
                                         episodes_file=c.EPISODES_CSV_FILENAME,
                                         summary_file=c.SUMMARY_CSV_FILENAME,
                                         median_fps=median_fps)

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
        filename = join(c.RESULTS_DIR, 'artifacts.json')
        with open(filename, 'w') as out_file:
            data = {'artifacts': {
                'mp4': anon(mp4_file),
                'performance_summary': anon(summary_file),
                'episodes': anon(episodes_file),
                'hdf5_observations': [anon(o) for o in hdf5_observations],
            }}
            json.dump(data, out_file, indent=2)
        log.info('Wrote artifacts inventory to %s' % anon(filename))
        latest_artifacts_filename = join(c.RESULTS_BASE_DIR,
                                         'latest-artifacts.json')
        shutil.copy2(filename, latest_artifacts_filename)
        print('\n****\nARTIFACTS INVENTORY COPIED TO: "%s"' +
              anon(latest_artifacts_filename))

    def num_unsaved_observations(self):
        return self.recorded_obz_count - self.num_saved_observations


def upload_artifacts(mp4_file:str, hdf5_observations: List[str]) \
        -> Tuple[str, str, str, List[str]]:
    youtube_id = utils.upload_to_youtube(mp4_file)
    if youtube_id:
        youtube_url = 'https://www.youtube.com/watch?v=%s' % youtube_id
        log.info('Successfully uploaded to YouTube! %s', youtube_url)
    else:
        youtube_url = ''
    mp4_url = upload_artifacts_to_s3([mp4_file], 'mp4')[0]
    hdf5_urls = upload_artifacts_to_s3(hdf5_observations, 'hdf5')
    return youtube_id, youtube_url, mp4_url, hdf5_urls


def upload_artifacts_to_s3(file_paths:List[str], directory:str) -> List[str]:
    from ue4helpers import AWSUtils
    ret = []
    for file_path in file_paths:
        s3path = 'artifacts/' + os.path.basename(c.RESULTS_DIR)
        key = s3path + ('/%s/' % directory) + os.path.basename(file_path)
        AWSUtils.upload_file(c.AWS_BUCKET, key=key, filename=file_path)
        ret.append('%s/%s' % (c.BUCKET_URL, key))
    return ret


def create_botleague_results(total_score: TotalScore, episode_scores, gist_url,
                             hdf5_observations, mp4_file, episodes_file,
                             summary_file, median_fps):
    ret = Box(default_box=True)
    ret.score = total_score.low
    ret.gist = gist_url

    ret.sensorimotor_specific.num_episodes = len(episode_scores)
    ret.sensorimotor_specific.median_fps = median_fps
    ret.sensorimotor_specific.num_steps = total_score.num_steps

    ret.driving_specific.max_gforce = total_score.max_gforce
    ret.driving_specific.max_kph = total_score.max_kph
    ret.driving_specific.trip_speed_kph = total_score.trip_speed_kph
    ret.driving_specific.collided_with_actor = total_score.collided_with_actor
    ret.driving_specific.collided_with_non_actor = \
        total_score.collided_with_non_actor
    ret.driving_specific.closest_vehicle_meters = \
        total_score.closest_vehicle_cm / 100
    ret.driving_specific.closest_vehicle_meters_while_at_least_4kph = \
        total_score.closest_vehicle_cm_while_at_least_4kph / 100
    ret.driving_specific.max_lane_deviation_meters = \
        total_score.max_lane_deviation_cm / 100

    # TODO: Closest distance to pedestrians

    # Add items to be uploaded by privileged code
    artifact_dir = c.BOTLEAGUE_RESULTS_DIR
    os.makedirs(artifact_dir, exist_ok=True)

    csv_relative_dir = 'csvs'

    summary_url, episodes_url = upload_artifacts_to_s3(
        [summary_file, episodes_file], csv_relative_dir)

    ret.problem_specific.summary = summary_url
    ret.problem_specific.episodes = episodes_url

    youtube_id, youtube_url, mp4_url, hdf5_urls = \
        upload_artifacts(mp4_file, hdf5_observations)

    ret.youtube = youtube_url
    ret.mp4 = mp4_url
    ret.problem_specific.hdf5_observations = hdf5_urls

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
      "max_step_milliseconds": 500.134323,
      "dropped_steps": 34,
  },

}
    """

    results_json_filename = join(artifact_dir, 'results.json')

    ret.to_json(filename=results_json_filename, indent=2)
    log.info('Wrote botleague results to %s' % results_json_filename)
    copy_dir_clean(src=c.BOTLEAGUE_RESULTS_DIR,
                   dest=c.LATEST_BOTLEAGUE_RESULTS)

    if c.BOTLEAGUE_CALLBACK:
        resp = requests.post(c.BOTLEAGUE_CALLBACK, data=ret.to_dict())
    else:
        raise RuntimeError('No botleague callback set, cannot post results!')

    return ret, resp


def make_needs_upload(base_dir:str, relative_dir:str, file:str,
                      upload_to:str) -> Box:
    ret = Box(default_box=True)
    upload = ret.needs_upload
    abs_dir = join(base_dir, relative_dir)
    os.makedirs(abs_dir, exist_ok=True)
    shutil.copy2(file, abs_dir)
    if relative_dir:
        upload.relative_path = '/'.join([relative_dir, basename(file)])
    else:
        upload.relative_path = basename(file)
    upload.upload_to = upload_to
    return ret
