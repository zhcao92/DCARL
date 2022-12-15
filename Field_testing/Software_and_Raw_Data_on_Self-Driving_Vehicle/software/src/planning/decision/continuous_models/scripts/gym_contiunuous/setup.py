from setuptools import setup

setup(name='carla_trainning',
      version='0.0.2',
      install_requires=['gym'],  # And any other dependencies foo needs
      packages=['gym_routing', 'gym_routing.envs']
)
