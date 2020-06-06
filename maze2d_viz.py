import os
import numpy as np
from utils.maze_viz import *
from utils.localization_viz import *
import matplotlib.pyplot as plt
import os

actions = [[False, 90], [False, -90], [True, 0]] # the action it chooses, -90 means turning left, 90 means turning right in VizDoom
# which is opposite to the refenrence in the Maze2D map. Thus, the actions list represents for [left, right, move forward]

class Maze2D(object):

    def __init__(self, args):
        self.args = args
        self.maze_idx = 0
        self.pid = os.getpid()
        return

    def game_init(self):
        game = DoomGame()
        game = set_doom_configurations(game, self.args)
        game.init()
        self.game = game

    def reset(self):
        # Load a test maze during evaluation
        if self.args.evaluate != 0:  # evalution == 0 means it is training now, need to generate a new map different from the existing ones
            index = self.maze_idx % 100
            self.game.set_doom_map("map{}".format(index))
            # this is for showing the map in a list
            MAP = []
            with open('{}{}_TEST_MAP{:02d}.txt'.format(self.args.sources, self.args.map_size, index + 100), 'r') as f:
                for line in f:
                    A_line = list(line.strip('\n'))
                    A_line = [float(x) if type(x) is str else None for x in A_line]
                    MAP.append(A_line)
            self.map_design = np.array(MAP) # should be a numpy array
            self.game.new_episode()
            state = self.game.get_state()
            vars = state.game_variables  # to check the agent's position and orientation
            if int(vars[2]) == 0:
                self.orientation = 0 # should be only a number
            elif int(vars[2]) == 90:
                self.orientation = 3
            elif int(vars[2]) == 180:
                self.orientation = 2
            else:
                self.orientation = 1

            Position_X = (vars[0] - 48) / 96
            Position_Y = (vars[1] - 48) / 96
            self.position = tuple([int(Position_Y), int(Position_X)]) # should be a tuple

            self.maze_idx += 1
        else: # the default wad file is 7_TRAIN.wad
            index = self.maze_idx % 100
            self.game.set_doom_map("map{}".format(index))
            # this is for showing the map in a list
            MAP = []
            with open('{}{}_TRAIN_MAP{:02d}.txt'.format(self.args.sources, self.args.map_size, index), 'r') as f:
                for line in f:
                    A_line = list(line.strip('\n'))
                    A_line = [float(x) if type(x) is str else None for x in A_line]
                    MAP.append(A_line)
            self.map_design = np.array(MAP)  # should be a numpy array
            self.game.new_episode()
            state = self.game.get_state()
            vars = state.game_variables  # to check the agent's position and orientation
            if int(vars[2]) == 0:
                self.orientation = 0  # should be only a number
            elif int(vars[2]) == 90:
                self.orientation = 3
            elif int(vars[2]) == 180:
                self.orientation = 2
            else:
                self.orientation = 1
            Position_X = (vars[0] - 48) / 96
            Position_Y = (vars[1] - 48) / 96
            self.position = tuple([int(Position_Y), int(Position_X)])  # should be a tuple

            self.maze_idx += 1

        # Pre-compute likelihoods of all observations on the map for efficiency
        self.likelihoods = get_all_likelihoods(self.map_design)

        # Get current observation and likelihood
        curr_depth = get_depth(self.map_design, self.position,
                               self.orientation)
        curr_likelihood = self.likelihoods[int(curr_depth) - 1]

        # Posterior is just the likelihood as prior is uniform
        self.posterior = curr_likelihood

        # Renormalization of the posterior
        self.posterior /= np.sum(self.posterior)

        # make the gif figures for a defense meeting
        if self.args.evaluate == 1:
            for o in state.objects:
                if o.name == "DoomPlayer":
                    print(o.position_x,o.position_y)
                    if int(vars[2]) == 0:
                        plt.plot(o.position_x, o.position_y, color='red', marker='>', markersize=4)
                    elif int(vars[2]) == 90:
                        plt.plot(o.position_x, o.position_y, color='red', marker='^', markersize=4)
                    elif int(vars[2]) == 180:
                        plt.plot(o.position_x, o.position_y, color='red', marker='<', markersize=4)
                    else:
                        plt.plot(o.position_x, o.position_y, color='red', marker='v', markersize=4)
            for s in state.sectors:
                for l in s.lines:
                    if l.is_blocking:
                        plt.plot([l.x1, l.x2], [l.y1, l.y2], color='black', linewidth=2)
            index_list = self.posterior.tolist()
            max_index = index_list.index(max(index_list))
            for y in range(0, self.args.map_size+2):
                for x in range(0, self.args.map_size+2):
                    if self.posterior[max_index, y, x] != 0:
                        plt.plot(x * 96 + 48, y * 96 + 48, color='green', marker='s',
                                 markersize=36, alpha=0.5 * self.posterior[max_index, y, x])
            if max_index == 0:
                plt.title("East")
            elif max_index == 1:
                plt.title("South")
            elif max_index == 2:
                plt.title("West")
            else:
                plt.title("North")
            plt.show()
        self.t = 0

        # next state for the policy model
        self.state = np.concatenate((self.posterior, np.expand_dims(
                                     self.map_design, axis=0)), axis=0)
        # if self.t == 0:
        #     print(self.pid, self.position, self.orientation)
        #     map = state.automap_buffer
        #     if map is not None:
        #         plt.imshow(map, cmap='gray')
        #         plt.title(self.pid)
        #         plt.show()

        return self.state, int(curr_depth)

    def step(self, action_id):
        state = self.game.get_state()
        vars1 = state.game_variables
        # Get the observation before taking the action
        curr_depth = get_depth(self.map_design, self.position,
                               self.orientation)

        # Posterior from last step is the prior for this step
        prior = self.posterior

        # Transform the prior according to the action taken
        prior = transition_function(prior, curr_depth, action_id)

        # perform the action in VizDoom
        action = actions[action_id]
        # print("action:", action_id, action)
        if curr_depth == 1 and action_id == 2:
            action = [False, 0]
        self.game.make_action(action)

        state = self.game.get_state()
        vars = state.game_variables

        # Calculate position and orientation after taking the action
        Pposition, Oorientation = get_next_state(
            self.map_design, self.position, self.orientation, action_id)

        if int(vars[2]) == 0:
            self.orientation = 0  # should be only a number
        elif int(vars[2]) == 90:
            self.orientation = 3
        elif int(vars[2]) == 180:
            self.orientation = 2
        else:
            self.orientation = 1
        Position_X = (vars[0] - 48) / 96
        Position_Y = (vars[1] - 48) / 96
        self.position = tuple([int(Position_Y), int(Position_X)])  # should be a tuple

        # Get the observation and likelihood after taking the action
        curr_depth = get_depth(
            self.map_design, self.position, self.orientation)
        curr_likelihood = self.likelihoods[int(curr_depth) - 1]
        # Posterior = Prior * Likelihood

        self.posterior = np.multiply(curr_likelihood, prior)

        # this is for detecting the errors
        # if np.sum(self.posterior) == 0:
        #     print(self.t, action, ":", vars1, "->", vars, self.position, self.orientation, Pposition, Oorientation
        #           , self.map_design[int(Position_Y)][int(Position_X)], "distance:", curr_depth)
        #     map = state.automap_buffer
        #     if map is not None:
        #         plt.imshow(map, cmap='gray')
        #         plt.show()

        # Renormalization of the posterior
        self.posterior /= np.sum(self.posterior)

        # make the gif figures for a defense meeting
        if self.args.evaluate == 1:
            for o in state.objects:
                if o.name == "DoomPlayer":
                    print(o.position_x,o.position_y)
                    if int(vars[2]) == 0:
                        plt.plot(o.position_x, o.position_y, color='red', marker='>', markersize=4)
                    elif int(vars[2]) == 90:
                        plt.plot(o.position_x, o.position_y, color='red', marker='^', markersize=4)
                    elif int(vars[2]) == 180:
                        plt.plot(o.position_x, o.position_y, color='red', marker='<', markersize=4)
                    else:
                        plt.plot(o.position_x, o.position_y, color='red', marker='v', markersize=4)
            for s in state.sectors:
                for l in s.lines:
                    if l.is_blocking:
                        plt.plot([l.x1, l.x2], [l.y1, l.y2], color='black', linewidth=2)
            index_list = self.posterior.tolist()
            max_index = index_list.index(max(index_list))
            index = int(int(vars[2])/90)
            if index == 1:
                index = 3
            elif index == 3:
                index = 1
            for y in range(0, self.args.map_size+2):
                for x in range(0, self.args.map_size+2):
                    if self.posterior[index, y, x] != 0:
                        print(self.posterior[index, y, x])
                        plt.plot(x * 96 + 48, y * 96 + 48, color='green', marker='s',
                                 markersize=36, alpha=0.5 * self.posterior[index, y, x])
            if max_index == 0:
                plt.title("step:{},estimate_orientation:East".format(self.t))
            elif max_index == 1:
                plt.title("step:{},estimate_orientation:South".format(self.t))
            elif max_index == 2:
                plt.title("step:{},estimate_orientation:West".format(self.t))
            else:
                plt.title("step:{},estimate_orientation:North".format(self.t))
            name = "./figures/localization{}.png".format(self.t)
            plt.savefig(name)
            plt.show()

        # Calculate the reward
        reward = self.posterior.max()

        self.t += 1
        if self.t == self.args.max_episode_length:
            is_final = True
        else:
            is_final = False

        # next state for the policy model
        self.state = np.concatenate(
            (self.posterior, np.expand_dims(
                self.map_design, axis=0)), axis=0)

        return self.state, reward, is_final, int(curr_depth)
