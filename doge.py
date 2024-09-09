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
from ctypes import c_double, c_uint
import numpy as np


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

def main():
  current_image = Array(c_uint, 64*64*3)
  current_image[:] = np.random.randint(0, 255, (64, 64, 3), dtype=np.int16).flatten()
  # See configs.yaml for all options.
  config = embodied.Config(dreamerv3.Agent.configs['defaults'])
  config = config.update({
      **dreamerv3.Agent.configs['size12m'],
      'logdir': f'{os.getcwd()}/logdir/20240823T052847-example',
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

  import sys
  
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
  
  video_width = video_height = 64
  video_writer = cv.VideoWriter("./videos/pybullet_world_model_dodging_event_camera.avi", cv.VideoWriter_fourcc(*"XVID"), 100, (video_width, video_height))


  event_camera_process = Process(target=run_event_process, args=(100, True, (125, 125, 125), (255, 255, 255), (0, 0, 0), current_image))
  event_camera_process.start()

  for i in range(2):
    start = time.time()
    action, _, state = agent.policy(obs, state, mode='eval')
    end = time.time()
    print(f"Policy deltatime init: {(end - start) * 1000:.2f} ms")
  

  solo12_controller = RealSolo12()
  solo12_controller.prepare_walking()

  
  def don_t_lose_connection():
    for i in range(666):
      solo12_controller.step([0, 0, 0])
    
  dont_lose_connection_thread = threading.Thread(target=don_t_lose_connection)
  dont_lose_connection_thread.start()


  for i in range(NR_STEPS):
      #obs["state"][0][-3:] = command
      if i == 0:
        obs["is_first"] = np.array([False])
      start = time.time()
      init_start = start
      action, _, state = agent.policy(obs, state, mode='eval')
      end = time.time()
      print(f"Policy deltatime: {(end - start) * 1000:.2f} ms")
      action["action"] = action["action"][0]
      print(action["action"])
      action["reset"] = False
      times = np.append(times, (end - start) * 1000) if i > 2 else times
      start = time.time()
      obs = solo12_controller.step(action["action"])
      end = time.time()
      print(f"Controller deltatime: {(end - start) * 1000:.2f} ms")
      image = np.array(current_image[:]).reshape((64, 64, 3))
      obs["image"] = image
      #video_writer.write(image)
      sanitize_obs(obs)
      end = time.time()
      deltatime = (end - init_start) * 1000
      print(f"Time for whole loop: {deltatime:.2f} ms")

  avg_time = np.mean(times)
  print(f"Average time encoder + policy (given already rescaled image): {avg_time:.2f} ms")
  video_writer.release()
  dont_lose_connection_thread.join()
  event_camera_process.join()

if __name__ == '__main__':
  main()
