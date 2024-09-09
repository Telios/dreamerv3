import warnings
from functools import partial as bind
import os
import sys
os.environ['MUJOCO_GL'] = 'egl'
import dreamerv3
import embodied
warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')
import cv2 as cv
#import external.quadruped_rl.event_camera as event_camera
from external.quadruped_rl.controller import RealSolo12
import threading
from multiprocessing import Process

import dv_processing as dv
from datetime import timedelta
from multiprocessing import Array, shared_memory
from ctypes import c_double, c_uint, c_bool
import numpy as np
import time


def run_event_process(fps, downscale, rgb_background, rgb_pos, rgb_neg, current_img):
    capture = dv.io.CameraCapture()
    if not capture.isEventStreamAvailable():
        raise RuntimeError("Input camera does not provide an event stream.")
        
    capture.setDVSBiasSensitivity(dv.io.CameraCapture.BiasSensitivity.Low)
    capture.setDVXplorerEFPS(dv.io.CameraCapture.DVXeFPS.EFPS_CONSTANT_500)

    fps = fps
    downscale = downscale
    visualizer = dv.visualization.EventVisualizer(capture.getEventResolution())
    visualizer.setBackgroundColor(rgb_background)
    visualizer.setPositiveColor(rgb_pos)
    visualizer.setNegativeColor(rgb_neg)

    def slicing_callback(events: dv.EventStore):
        img = visualizer.generateImage(events)
        if downscale:
            img = cv.resize(img, (85, 64), interpolation=cv.INTER_NEAREST)
            img = img[0:64, 10:74]
        current_img[:] = np.array(img).flatten()
        
    slicer = dv.EventStreamSlicer()
    slicer.doEveryTimeInterval(timedelta(milliseconds=1000/fps), slicing_callback)

    while capture.isRunning():
        events = capture.getNextEventBatch()

        if events is not None:
            slicer.accept(events)

def run_controller_process(action, q_mes, accelerometer, angular_velocity, last_action, started):
    solo12_controller = RealSolo12()
    started.value = solo12_controller.started
    solo12_controller.prepare_walking()

    frequency = 100
    period = 1.0 / frequency
    while True:
        start_time = time.time()
        obs = solo12_controller.step(action[:])
        state = obs["state"]
        q_mes[:] = state[0:12]
        accelerometer[:] = state[12:15]
        angular_velocity[:] = state[15:18]
        last_action[:] = state[18:21]

        elapsed_time = time.time() - start_time
        sleep_time = period - elapsed_time
        if sleep_time > 0:
            time.sleep(sleep_time)
        end_time = time.time()
        print(f"Controller loop time: {(end_time - start_time) * 1000:.2f} ms")
    
def main():
  # Shared variables
  current_image = Array(c_uint, 64*64*3)
  current_image[:] = np.random.randint(0, 255, (64, 64, 3), dtype=np.int16).flatten()

  action = Array(c_double, 3)
  action[:] = np.array([0.0, 0.0, 0.0])

  started = Array(c_bool, 1)
  started.value = False


  # state
  q_mes = Array(c_double, 12)
  q_mes[:] = np.zeros(12)
  accelerometer = Array(c_double, 3)
  accelerometer[:] = np.zeros(3)
  angular_velocity = Array(c_double, 3)
  angular_velocity[:] = np.zeros(3)
  last_action = Array(c_double, 3)
  last_action[:] = np.zeros(3)

  # See configs.yaml for all options.
  config = embodied.Config(dreamerv3.Agent.configs['defaults'])
  config = config.update({
      **dreamerv3.Agent.configs['size12m'],
      'logdir': f'{os.getcwd()}/logdir/20240827T055930-example',
      'run.train_ratio': 512,
      'run.steps': 6e5,
      'enc.spaces': 'image|state',
      'dec.spaces': 'image|state',
      'run.script': 'eval_only',
      'replay_length': 65,
      'batch_size': 1,
      'replay_length_eval': 33,
      'run.log_video_fps': 50,
  })
  config = embodied.Flags(config).parse()

  print('Logdir:', config.logdir)
  logdir = embodied.Path(config.logdir)
  step = embodied.Counter()
  logger = embodied.Logger(step, [
      embodied.logger.TerminalOutput(),
      embodied.logger.JSONLOutput(logdir, 'metrics.jsonl'),
      embodied.logger.TensorBoardOutput(logdir),
      # embodied.logger.WandBOutput(logdir.name, config),
      # embodied.logger.MLFlowOutput(logdir.name),
  ])
  
  obs_space = {
    'image': embodied.Space(dtype=np.float32, shape=(64, 64, 3), low=0, high=255),
    'state': embodied.Space(dtype=np.float64, shape=(21,)),
    'is_first': embodied.Space(bool),
    'is_last': embodied.Space(bool),
    'reward': embodied.Space(np.float32),
    'is_terminal': embodied.Space(bool),
  }
  act_space = {
    'action': embodied.Space(np.float32, (3,), -1.0, 1.0),
    'reset': embodied.Space(dtype=bool),
  }
  agent = dreamerv3.Agent(obs_space, act_space, config)
  checkpoint = embodied.Checkpoint()
  checkpoint.agent = agent
  checkpoint.load(f"{config.logdir}/checkpoint.ckpt", keys=['agent'])
  
  NR_STEPS = 30 * 100 # seconds * 100, since 1 step = 10 ms
  obs = {}
  obs["image"] = np.zeros((1, 64, 64, 3), dtype=np.uint8)
  obs["state"] = np.zeros((1, 21), dtype=np.float64)
  obs["is_first"] = np.array([True])
  obs["is_last"] = np.array([False])
  obs["is_terminal"] = np.array([False])
  obs["reward"] = np.array([0.0])
  state = agent.init_policy(batch_size=1)
  import time
  times = np.array([])
  
  def sanitize_obs(obs):
    for key in obs.keys():
      if key == "state":
        obs[key] = obs[key].reshape((1, -1))
      elif key == "image":
        obs[key] = np.expand_dims(obs[key], axis=0)
      else:
        obs[key] = np.array([obs[key]])
    obs["is_first"] = np.array([False])
    obs["is_last"] = np.array([False])
    obs["is_terminal"] = np.array([False])
    obs["reward"] = np.array([0.0])

  def build_obs(q_mes, accelerometer, angular_velocity, last_action, image):
    obs = {}
    obs["image"] = np.array(image).reshape((64, 64, 3))
    obs["state"] = np.concatenate((q_mes, accelerometer, angular_velocity, last_action))
    obs["is_first"] = np.array([False])
    obs["is_last"] = np.array([False])
    obs["is_terminal"] = np.array([False])
    obs["reward"] = np.array([0.0])
    return obs
  
  video_width = video_height = 64
  video_writer = cv.VideoWriter("./videos/pybullet_world_model_dodging_event_camera.avi", cv.VideoWriter_fourcc(*"XVID"), 100, (video_width, video_height))


  event_camera_process = Process(target=run_event_process, args=(100, True, (125, 125, 125), (255, 255, 255), (0, 0, 0), current_image))
  event_camera_process.start()
  controller_process = Process(target=run_controller_process, args=(action, q_mes, accelerometer, angular_velocity, last_action, started))
  controller_process.start()

  while not started.value:
    time.sleep(1)

  for i in range(NR_STEPS):
      #obs["state"][0][-3:] = command
      if i == 1:
        obs["is_first"] = np.array([False])
      start = time.time()
      init_start = start
      action_policy, _, state = agent.policy(obs, state, mode='eval')
      end = time.time()
      print(f"Policy deltatime: {(end - start) * 1000:.2f} ms")
      action_policy["action"] = action_policy["action"][0]
      print(action_policy["action"])
      action[:] = action_policy["action"]
      action_policy["reset"] = False
      times = np.append(times, (end - start) * 1000) if i > 2 else times
      obs = build_obs(q_mes[:], accelerometer[:], angular_velocity[:], last_action[:], current_image[:])
      #video_writer.write(image)
      sanitize_obs(obs)
      end = time.time()
      deltatime = (end - init_start) * 1000
      print(f"Time for whole loop: {deltatime:.2f} ms")

  avg_time = np.mean(times)
  print(f"Average time encoder + policy (given already rescaled image): {avg_time:.2f} ms")
  video_writer.release()
  event_camera_process.join()
  controller_process.join()

if __name__ == '__main__':
  main()
