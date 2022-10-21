from turtle import shape
import numpy as np
import gym
from utils import dynamics_eq
from gym import spaces
from gym.utils import seeding

class Pendrogone(gym.Env):
    # LIMITS = np.array([2.5, 2.5])
    LIMITS = np.array([15, 15])
    T = 0.02

    metadata = {
        'render.modes': ['human'],
        'video.frames_per_second' : 1/T
    }

    # def __init__(self, gravity=9.807, q_mass=2.5, pend_length=1.0, Ixx=1.0):
    #     self.gravity = gravity #: [m/s2] acceleration


    def __init__(self, g=9.807, mQ=2.5, l=1.0, IQ=1.0, as_numpy=False):
        # Dynamics constants
        # yapf: disable
        self.g = g           # gravity (m / s**2)
        self.mQ = mQ         # mass (kg)
        self.l = l           # half-length (m)
        self.IQ = IQ         # moment of inertia about the out-of-plane axis (kg * m**2)
        self.as_numpy = as_numpy
        # yapf: enable
        
        # Pendulum
        self.mp = self.mQ*4
        self.L = self.l*2
        self.Ip = self.mp*(self.L/2)**2
        
        self.ddx_func, self.ddy_func, self.ddθ_func, self.ddϕ_func, self.dfds_func, self.dfdu_func = dynamics_eq()
        
        self.state_dim = 8
        self.control_dim = 2
        # ## Quadrotor stuff
        # self.q_mass = q_mass #: [kg] mass
        # self.Ixx = Ixx
        # self.arm_length =  # [m]
        # self.arm_width = 0.02 # [m]
        # self.height = 0.02 # [m]

        # limits
        self.limits = Pendrogone.LIMITS
        # self.q_maxAngle = np.pi / 2

        # ## Load stuff
        # self.l_mass = self.q_mass*4
        # self.cable_length = 0.82
        # self.cable_width = 0.01
        # self.l_maxAngle = np.pi / 3

        self.Mass = self.mQ + self.mp

        # max and min force for each motor
        self.maxU = self.Mass * self.g
        self.minU = 0
        self.dt = Pendrogone.T

        """
        The state has 8 dimensions:
         xm,zm :quadrotor position
         phi :quadrotor angle
         theta :load angle, for now the tension in the cable 
                is always non zero
        """

        high = np.array([
            np.finfo(np.float32).max, # xl
            np.finfo(np.float32).max, # zl

            1.0, # Sth
            1.0, # Cth
            1.0, # Sphi
            1.0, # Cphi

            # np.finfo(np.float32).max, # th
            # np.finfo(np.float32).max, # phi


            np.finfo(np.float32).max, # xl_dot
            np.finfo(np.float32).max, # zl_dot
            np.finfo(np.float32).max, # th_dot
            np.finfo(np.float32).max, # phi_dot
        ])
        
        self.action_space = spaces.Box(
            low = np.array([self.minU, self.minU]),
            high = np.array([self.maxU, self.maxU]),
            dtype = np.float32
        )
        self.observation_space = spaces.Box(
            low = -high,
            high = high,
            dtype = np.float32
        )

        self.seed()
        self.viewer = None
        # Need to always call the render() method
        self.state = None # yet :v




    def _apply_action(self, control):
        x, y, θ, ϕ, dx, dy, dθ, dϕ = self.state
        clipped_T = np.clip(control, self.action_space.low, self.action_space.high)
        
        T1, T2 = clipped_T

        ds = [
            dx,
            dy,
            dθ,
            dϕ,
            self.ddx_func(self.Ip, self.mp, self.L, ϕ, dϕ, self.IQ, self.mQ, self.l, θ, dθ, T1, T2, self.g),
            self.ddy_func(self.Ip, self.mp, self.L, ϕ, dϕ, self.IQ, self.mQ, self.l, θ, dθ, T1, T2, self.g),
            self.ddθ_func(self.Ip, self.mp, self.L, ϕ, dϕ, self.IQ, self.mQ, self.l, θ, dθ, T1, T2, self.g),
            self.ddϕ_func(self.Ip, self.mp, self.L, ϕ, dϕ, self.IQ, self.mQ, self.l, θ, dθ, T1, T2, self.g),
        ]
        ds = np.array(ds)
        
        self.state = ds * self.dt + self.state

        return clipped_T

    def random_uniform(self, low, high):
        assert high.shape == low.shape
        
        width = high - low
        return width*np.random.rand(*high.shape) + low
    
    
    def reset(self):
        """
        Set a random objective position for the load
        sampling a position for the quadrotor and then
        calculating the load position
        """
        l_pos = self.limits - self.L
        pos_drone = self.random_uniform(low=-l_pos, high=l_pos)

        
        l_angles = np.array([ 0.1, 0.4 ])
        angles = np.pi + self.random_uniform(low=-l_angles, high=l_angles)
        # angle = [ 0.0, 0.0 ]

        self.state = np.array([
            pos_drone[0],
            pos_drone[1],
            angles[0],
            angles[1],
            0, 0, 0, 0
        ])
        # self.objective = np.array([0.0, 0.0, 0.0, np.pi, 0.0,0.0,0.0,0.0])
        self.objective = np.array([0.0, 0.0])

        self.specific_reset()

        return self.obs

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    @staticmethod
    def transform(x0, angle, xb):
        T = np.array([ [np.cos(angle), -np.sin(angle)],
                       [np.sin(angle),  np.cos(angle)] ])
        return x0 + T.dot(xb)

    @property
    def pos_quad(self):
        """
        Quadrotor position in the inertial frame
        """
        return self.state[0:2] + self.L * self.p

    @property
    def p(self):
        """
        unit vector from quadrotor to the load
        """
        return np.array([ np.sin(self.state[3]), -np.cos(self.state[3]) ])

    @property
    def potential(self):
        dist = np.linalg.norm([ self.state[0] - self.objective[0],
                                self.state[1] - self.objective[1]] )

        return - dist

    def alive_bonus(self):
        # dead = np.absolute(self.state[2]) > self.q_maxAngle \
        #     or np.absolute(self.state[3]) > self.l_maxAngle \
        #     or np.absolute(self.state[0]) > Pendrogone.LIMITS[0] \
        #     or np.absolute(self.state[1]) > Pendrogone.LIMITS[1]

        dead = np.absolute(self.state[0]) > Pendrogone.LIMITS[0] \
            or np.absolute(self.state[1]) > Pendrogone.LIMITS[1]

        return -200 if dead else +0.5

    def orientation_bonus(self):
        """
        Reward for upright drone (θ=0) and penalize for dθ != 0
        """
        _, _, θ, _, _, _, dθ, _ = self.state
        return 10 * np.cos(θ) - 0.1 * dθ
    
    def pendulum_bonus(self):
        """
        Reward for upright pendulum (ϕ=pi) and penalize for dϕ != 0
        """
        _, _, _, ϕ, _, _, _, dϕ = self.state
        return 100 * (-np.cos(ϕ)) - 10 * dϕ

    @property
    def obs(self):
        x, y, phi, th, x_dot, y_dot, phi_dot, th_dot = self.state

        obj_x, obj_y = self.objective
        
        obs = np.array([
            x - obj_x, y - obj_y,
            np.sin(phi), np.cos(phi),
            np.sin(th), np.cos(th),
            x_dot, y_dot,
            phi_dot, th_dot,
        ])

        return obs
    
    def render(self, mode='human'):
        from gym.envs.classic_control import rendering
        screen_width = 800
        screen_height = 800

        xq,zq,phi,theta = self.state[0:4]
        xl,zl = self.pos_quad

        t1_xy = Pendrogone.transform(self.pos_quad,
                                     self.state[2],
                                     np.array([self.l, 0]))
        t2_xy = Pendrogone.transform(self.pos_quad,
                                     self.state[2],
                                     np.array([-self.l, 0]))
        tl_xy = self.state[0:2]

        to_xy = self.objective

        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)
            self.viewer.set_bounds(-Pendrogone.LIMITS[0], Pendrogone.LIMITS[0],
                                   -Pendrogone.LIMITS[1], Pendrogone.LIMITS[1])

            # Render quadrotor frame
            ql,qr,qt,qb = -self.l, self.l, 0.2, -0.2
            self.frame_trans = rendering.Transform(rotation=phi, translation=(xq,zq))
            frame = rendering.FilledPolygon([(ql,qb), (ql,qt), (qr,qt), (qr,qb)])
            frame.set_color(0, .8, .8)
            frame.add_attr(self.frame_trans)
            self.viewer.add_geom(frame)

            # Render load cable
            ll,lr,lt,lb = -0.05, 0.05, 0, -self.L
            self.cable_trans = rendering.Transform(rotation=theta, translation=(xq,zq))
            cable = rendering.FilledPolygon([(ll,lb), (ll,lt), (lr,lt), (lr,lb)])
            cable.set_color(.1, .1, .1)
            cable.add_attr(self.cable_trans)
            self.viewer.add_geom(cable)

            # Render right propeller
            self.t1_trans = rendering.Transform(translation=t1_xy)
            thruster1 = self.viewer.draw_circle(.04)
            thruster1.set_color(.8, .8, 0)
            thruster1.add_attr(self.t1_trans)
            self.viewer.add_geom(thruster1)

            # Render left propeller 
            self.t2_trans = rendering.Transform(translation=t2_xy)
            thruster2 = self.viewer.draw_circle(.04)
            thruster2.set_color(.8, .8, 0)
            thruster2.add_attr(self.t2_trans)
            self.viewer.add_geom(thruster2)

            # Render load
            self.tl_trans = rendering.Transform(translation=tl_xy)
            load = self.viewer.draw_circle(.08)
            load.set_color(.8, .3, .8)
            load.add_attr(self.tl_trans)
            self.viewer.add_geom(load)

            # Render objective point
            self.to_trans = rendering.Transform(translation=to_xy)
            objective = self.viewer.draw_circle(.02)
            objective.set_color(1., .01, .01)
            objective.add_attr(self.to_trans)
            self.viewer.add_geom(objective)

            
        self.frame_trans.set_translation(xq,zq)
        self.frame_trans.set_rotation(phi)
        self.cable_trans.set_translation(xq,zq)
        self.cable_trans.set_rotation(theta)
        
        self.t1_trans.set_translation(t1_xy[0], t1_xy[1])
        self.t2_trans.set_translation(t2_xy[0], t2_xy[1])
        self.tl_trans.set_translation(tl_xy[0], tl_xy[1])

        self.to_trans.set_translation(to_xy[0], to_xy[1])
        
        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer: self.viewer.close()

    def step(self, action):
        # This first block of code should not change
        old_potential = self.potential

        
        self._apply_action(action)
        
        potential = self.potential

        
        control_r = -0.07 * np.array(action).dot(np.array(action))
        alive_r = float(self.alive_bonus())
        pot_r = 50 * (potential - old_potential)

        # -potential = distance to the objective
        shape_r = Pendrogone.reward_shaping( -potential,
                                                  np.linalg.norm(self.state[4:6]) )

        ori_r = self.orientation_bonus()
        pendulum_r = self.pendulum_bonus()
        
        reward = -np.linalg.norm(self.state[:2])/15 - np.linalg.norm(self.state[2:4] - np.array([0,np.pi])) - np.linalg.norm(self.state[4:6])/15 - np.linalg.norm(self.state[6:])

        # reward = np.tanh(1 - 0.0005*(abs(self.state - np.array([0,0,0,np.pi,0,0,0,0]))).sum())

        # r = np.array([control_r, alive_r, ori_r, pendulum_r, pot_r, shape_r]).round(2)
        # print(r)
        # reward = alive_r - np.linalg.norm(self.state[:2]) - np.linalg.norm(self.state[2:] - np.array([0,np.pi,0,0,0,0]))
        
        done = alive_r < 0

        return self.obs, reward, done, {}
    
    # def get_reward(self):
    #     """Uses current pose of sim to return reward."""
    #     #reward = 1-(0.3*(abs(self.sim.pose[:3] - self.target_pos))).sum()
    #     #reward = np.tanh(reward)
    #     reward = 
    #     return reward

    # @staticmethod
    # def reward_shaping(dist, vel):
    #     # print(dist, vel)
    #     c = 5
    #     dist_r = np.exp(- np.abs(3.5*dist)**2)
    #     vel_r = np.power(np.exp(- np.abs(vel)), np.exp(- 2.5 * np.abs(dist)))
    #     # vel_r = np.exp(- np.abs(vel))
    #     # mask = (dist <= 0.2) * (vel > 1)
    #     result = c * dist_r * vel_r

    #     # return np.logical_not(mask) * result + mask * -3.0
    #     return result

    @staticmethod
    def reward_shaping(dist, vel):
        # print(dist, vel)
        c = 5
        dist_r = np.exp(- 2 * dist)
        return c * dist_r

        # c1 = 10
        # c2 = 0.2
        # dist_r = -c1 * np.tanh(c2 * dist)
        # return dist_r
    

    @staticmethod
    def normal_dist(mu, sigma_2):
        c = 1/np.sqrt(2*np.pi*sigma_2)

        return lambda x: c * np.exp(- (x-mu)**2 / (2*sigma_2))

    @staticmethod
    def exponential():
        # return lambda d, v : max(3 - (3*d) ** 0.4, 0.0) * \
        #     (4 - min(4, max(v, 0.001)))/4 ** (1/max(0.1, d))
        # return lambda d : 3*max(1 - (d/4) ** 0.4, 0.0)
        return lambda d : np.exp(- np.abs(d*2))

    def specific_reset(self):
        pass
