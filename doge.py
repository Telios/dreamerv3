import warnings
from functools import partial as bind
import os
import sys
#sys.path.append(os.path.join(os.getcwd(), '.'))
os.environ['MUJOCO_GL'] = 'egl'
import dreamerv3
import embodied
warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')
import cv2 as cv
from utils.event_camera import EventCamera
from external.quadruped_rl.controller import RealSolo12
import threading

def main():

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

  import numpy as np
  import sys
  sys.path.append("../")
  from testing.envs.solo12_pybullet import Solo12Env
  from embodied.envs import from_gym
  env = Solo12Env(max_steps=5000)
  env = from_gym.FromGym(env, obs_key='image')
  env_dreamer = dreamerv3.wrap_env(env, config)
  
  agent = dreamerv3.Agent(env_dreamer.obs_space, env_dreamer.act_space, config)
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
  
  video_width = video_height = 64
  video_writer = cv.VideoWriter("./videos/pybullet_world_model_dodging_event_camera.avi", cv.VideoWriter_fourcc(*"XVID"), 100, (video_width, video_height))

  event_camera = EventCamera(downscale=True)
  event_camera_thread = threading.Thread(target=event_camera.run)
  event_camera_thread.start()

  solo12_controller = RealSolo12()
  solo12_controller.prepare_walking()

  for i in range(NR_STEPS):
      #obs["state"][0][-3:] = command
      if i == 1:
        obs["is_first"] = np.array([False])
      start = time.time()
      action, _, state = agent.policy(obs, state, mode='eval')
      end = time.time()
      action["action"] = action["action"][0]
      action["reset"] = False
      times = np.append(times, (end - start) * 1000) if i > 2 else times
      obs = solo12_controller.step(action["action"])
      image = event_camera.current_image
      obs["image"] = image
      video_writer.write(image)
      sanitize_obs(obs)
      end = time.time()
      deltatime = (end - start) * 1000
      print(f"Time for whole loop: {deltatime:.2f} ms")

  avg_time = np.mean(times)
  print(f"Average time encoder + policy (given already rescaled image): {avg_time:.2f} ms")
  video_writer.release()
  event_camera_thread.join()

if __name__ == '__main__':
  main()
