import warnings
from functools import partial as bind
import os
os.environ['MUJOCO_GL'] = 'egl'
import dreamerv3
import embodied
warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')
import cv2 as cv


def main():

  # See configs.yaml for all options.
  config = embodied.Config(dreamerv3.Agent.configs['defaults'])
  config = config.update({
      **dreamerv3.Agent.configs['size12m'],
      'logdir': f'{os.getcwd()}/logdir/20240703T055543-example',
      'run.train_ratio': 512,
      'run.steps': 6e5,
      'enc.spaces': 'image|state',
      'dec.spaces': 'image|state',
      'run.script': 'eval_only',
      'replay_length': 65,
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
  from testing.envs.solo12_v1 import Solo12Env
  from embodied.envs import from_gym
  env = Solo12Env(xml_file=f"{os.getcwd()}/../testing/assets/scene.xml", render_mode="rgb_array", width=64, height=64)
  env = from_gym.FromGym(env, obs_key='image')
  env_dreamer = dreamerv3.wrap_env(env, config)
  
  agent = dreamerv3.Agent(env_dreamer.obs_space, env_dreamer.act_space, config)
  checkpoint = embodied.Checkpoint()
  checkpoint.agent = agent
  checkpoint.load(f"{config.logdir}/checkpoint.ckpt", keys=['agent'])
  
  NR_STEPS = 1000
  obs = {}
  obs["image"] = np.zeros((1, 64, 64, 3), dtype=np.uint8)
  obs["state"] = np.zeros((1, 37), dtype=np.float64)
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
  
  video_writer = cv.VideoWriter("./videos/event_based_world_model_dodging.avi", cv.VideoWriter_fourcc(*"XVID"), 30, (64, 64))
  command = np.array([1.0, 0.0, 0.0])
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
      obs = env.step(action)
      image = cv.cvtColor(env._env.render(camera_id=1), cv.COLOR_RGB2BGR)
      video_writer.write(image)
      sanitize_obs(obs)
  avg_time = np.mean(times)
  print(f"Average time encoder + policy (given already rescaled image): {avg_time:.2f} ms")

if __name__ == '__main__':
  main()
