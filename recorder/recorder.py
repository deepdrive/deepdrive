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
            server_public_run = bool(c.BOTLEAGUE_CALLBACK)
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
                gist_url = None

            hdf5_observations = glob.glob(c.HDF5_SESSION_DIR + '/*.hdf5')
            self.create_artifacts_inventory(
                hdf5_observations=hdf5_observations,
                episodes_file=c.EPISODES_CSV_FILENAME,
                summary_file=c.SUMMARY_CSV_FILENAME,
                mp4_file=mp4_file)

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
    if 'UPLOAD_TO_YOUTUBE' in os.environ:
        youtube_id = utils.upload_to_youtube(mp4_file)
        if youtube_id:
            youtube_url = 'https://www.youtube.com/watch?v=%s' % youtube_id
            log.info('Successfully uploaded to YouTube! %s', youtube_url)
        else:
            youtube_url = ''
    else:
        youtube_url = ''
        youtube_id = ''
    mp4_url = upload_artifacts_to_s3([mp4_file], 'mp4')[0]
    hdf5_urls = upload_artifacts_to_s3(hdf5_observations, 'hdf5')
    return youtube_id, youtube_url, mp4_url, hdf5_urls


def upload_artifacts_to_s3(file_paths:List[str], directory:str,
                           use_gcp=True) -> List[str]:
    from ue4helpers import GCPUtils, AWSUtils
    ret = []
    for file_path in file_paths:
        s3path = 'artifacts/' + os.path.basename(c.RESULTS_DIR)
        key = s3path + ('/%s/' % directory) + os.path.basename(file_path)
        if use_gcp:
            bucket = c.GCP_BUCKET
            url = c.GCP_BUCKET_URL
            storage_utils = GCPUtils
        else:
            bucket = c.AWS_BUCKET
            url = c.AWS_BUCKET_URL
            storage_utils = AWSUtils
        storage_utils.upload_file(bucket, key=key, filename=file_path)
        ret.append('%s/%s' % (url, key))
    return ret


def create_botleague_results(total_score: TotalScore, episode_scores, gist_url,
                             hdf5_observations, mp4_file, episodes_file,
                             summary_file, median_fps):
    ret = Box(default_box=True)
    ts = total_score
    if gist_url:
        ret.gist = gist_url

    def sum_over_episodes(key):
        return sum(getattr(e, key) for e in episode_scores)

    ret.sensorimotor_specific = get_sensorimotor_specific_results(
        episode_scores, median_fps, sum_over_episodes, ts)
    total_time = ret.sensorimotor_specific.total_episode_seconds
    ret.driving_specific, ds_score = get_driving_specific_results(
        episode_scores, sum_over_episodes, total_time, ts)
    ret.score = ds_score
    artifact_dir = c.BOTLEAGUE_RESULTS_DIR
    os.makedirs(artifact_dir, exist_ok=True)
    csv_relative_dir = 'csvs'
    if c.BOTLEAGUE_CALLBACK or c.UPLOAD_RESULTS:
        create_uploaded_artifacts(csv_relative_dir, episodes_file,
                                  hdf5_observations, mp4_file, ret,
                                  summary_file)
    else:
        use_local_artifacts(episodes_file, hdf5_observations, mp4_file, ret,
                            summary_file)
    store_results(artifact_dir, ret)
    resp = send_results(ret)

    return ret, resp


def send_results(ret):
    if c.BOTLEAGUE_CALLBACK:
        resp = requests.post(c.BOTLEAGUE_CALLBACK, data=ret.to_dict())
    else:
        resp = None
    return resp


def store_results(artifact_dir, ret):
    results_json_filename = join(artifact_dir, 'results.json')
    ret.to_json(filename=results_json_filename, indent=2)
    log.info('Wrote botleague results to %s' % results_json_filename)
    copy_dir_clean(src=c.BOTLEAGUE_RESULTS_DIR,
                   dest=c.LATEST_BOTLEAGUE_RESULTS)


def get_sensorimotor_specific_results(episode_scores, median_fps,
                                      sum_over_episodes, ts) -> Box:
    ret = Box()
    ret.num_episodes = len(episode_scores)
    ret.median_fps = median_fps
    ret.num_steps = ts.num_steps
    ret.total_episode_seconds = sum_over_episodes('episode_time')
    return ret


def use_local_artifacts(episodes_file, hdf5_observations, mp4_file, ret,
                        summary_file):
    ret.mp4 = mp4_file
    ret.problem_specific.hdf5_observations = hdf5_observations
    ret.problem_specific.summary = summary_file
    ret.problem_specific.episodes = episodes_file


def create_uploaded_artifacts(csv_relative_dir, episodes_file,
                              hdf5_observations, mp4_file, ret, summary_file):
    summary_url, episodes_url = upload_artifacts_to_s3(
        [summary_file, episodes_file], csv_relative_dir)
    ret.problem_specific.summary = summary_url
    ret.problem_specific.episodes = episodes_url
    youtube_id, youtube_url, mp4_url, hdf5_urls = \
        upload_artifacts(mp4_file, hdf5_observations)
    ret.youtube = youtube_url
    ret.mp4 = mp4_url
    ret.problem_specific.hdf5_observations = hdf5_urls


def get_driving_specific_results(episode_scores, sum_over_episodes,
                                 total_time, ts):
    # TODO: Closest distance to pedestrians
    # See https://docs.google.com/spreadsheets/d/1Nm7_3vUYM5pIs2zLWM2lO_TCoIlVpwX-YRLVo4S4-Cc/edit#gid=0
    #   for balancing score coefficients

    ret = Box()
    score = 0
    ret.max_gforce = ts.max_gforce
    ret.uncomfortable_gforce_seconds = sum_over_episodes(
        'uncomfortable_gforce_seconds')
    ret.jarring_gforce_seconds = \
        sum_over_episodes('jarring_gforce_seconds')
    ret.harmful_gforces = \
        any(e.harmful_gforces for e in episode_scores)
    ret.comfort_pct = 100 - ret.uncomfortable_gforce_seconds / total_time * 100
    score -= ret.comfort_pct * 100
    ret.jarring_pct = ret.jarring_gforce_seconds / total_time * 100
    score -= ret.jarring_pct * 500
    ret.max_gforce = ts.max_gforce
    ret.max_kph = ts.max_kph
    ret.trip_speed_kph = ts.trip_speed_kph
    score += ts.trip_speed_kph * 10
    ret.collided_with_vehicle = ts.collided_with_vehicle
    if ts.collided_with_vehicle:
        score -= 1e4
    ret.collided_with_non_actor = ts.collided_with_non_actor
    if ts.collided_with_non_actor:
        score -= 2.5e3
    ret.closest_vehicle_meters = ts.closest_vehicle_cm / 100
    ret.closest_vehicle_meters_while_at_least_4kph = \
        ts.closest_vehicle_cm_while_at_least_4kph / 100
    ret.max_lane_deviation_meters = ts.max_lane_deviation_cm / 100
    return ret, score


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
