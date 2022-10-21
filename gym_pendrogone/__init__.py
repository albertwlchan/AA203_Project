from gym.envs.registration import register

register(
    id='Pendrogone-v0',
    entry_point='gym_pendrogone.envs:Pendrogone',
    max_episode_steps=200
)


register(
    id='Pendrogone-v1',
    entry_point='gym_pendrogone.envs:Pendrogone_zero',
    max_episode_steps=200
)