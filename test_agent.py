import warnings
from functools import partial as bind
import os
os.environ['MUJOCO_GL'] = 'egl'
import dreamerv3
import embodied
warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')


def main():

  # See configs.yaml for all options.
  config = embodied.Config(dreamerv3.Agent.configs['defaults'])
  config = config.update({
      **dreamerv3.Agent.configs['size12m'],
      'logdir': f'{os.getcwd()}/logdir/{embodied.timestamp()}-example',
      'run.train_ratio': 512,
      'run.steps': 6e5,
      'enc.spaces': 'image|state',
      'dec.spaces': 'image|state',
      'run.script': 'train',
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
  from dm_control import suite
  from dreamerv3.embodied.envs.dmc import DMC
  from dreamerv3 import embodied

  env = suite.load('walker', 'walk')
  env = DMC(env)
  
  agent = dreamerv3.Agent(env.obs_space, env.act_space, step, config)
  checkpoint = embodied.Checkpoint()
  checkpoint.agent = agent
  checkpoint.load(config.run.from_checkpoint, keys=['agent'])

  
  # dummy observations
  observations = []
  obs = {}
  for i in range(10000):
      obs["image"] = np.random.randint(0, 256, (1, 64, 64, 3))
      obs["is_first"] = np.array([True]) if i == 0 else np.array([False])
      obs["is_terminal"] = np.array([False])
      observations.append(obs)
  
  state = None
  times = np.array([])
  for obs in tqdm(observations):
      start = time.time()
      action, state = agent.policy(obs, state, mode='eval')
      end = time.time()
      times = np.append(times, (end - start) * 1000)
  avg_time = np.mean(times)
  print(f"Average time enocer + policy (given already rescaled image): {avg_time:.2f} ms")




if __name__ == '__main__':
  main()
