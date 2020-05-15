#!/usr/bin/env python3

#####################################################################
# an example borrowed from https://github.com/mwydmuch/ViZDoom/blob/master/examples/python/basic.py
# this implements the most fundamental tasks in localization problems
#####################################################################

from __future__ import print_function
import vizdoom as vzd
import cv2
import matplotlib.pyplot as plt
import numpy as np
import copy

from random import choice
from time import sleep

if __name__ == "__main__":
    # Create DoomGame instance. It will run the game and communicate with you.
    game = vzd.DoomGame()

    # Now it's time for configuration!
    # load_config could be used to load configuration instead of doing it here with code.
    # If load_config is used in-code configuration will also work - most recent changes will add to previous ones.
    # game.load_config("../../scenarios/basic.cfg")

    # Sets path to additional resources wad file which is basically your scenario wad.
    # If not specified default maps will be used and it's pretty much useless... unless you want to play good old Doom.
    # game.set_doom_scenario_path("../../scenarios/basic.wad")
    game.set_doom_scenario_path("../../VizDoom/7_TRAIN.wad")
    # game.set_doom_scenario_path("../../../vizdoomgymmaze/vizdoomgymmaze/envs/scenarios/four/four_1.wad")

    # Sets map to start (scenario .wad files can contain many maps).
    # game.set_doom_map("map00")

    # print(map_array)

    # Sets resolution. Default is 320X240
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)

    # Sets the screen buffer format. Not used here but now you can change it. Default is CRCGCB.
    game.set_screen_format(vzd.ScreenFormat.RGB24)

    # Enables depth buffer.
    game.set_depth_buffer_enabled(True)

    # Enables labeling of in game objects labeling.
    game.set_labels_buffer_enabled(True)

    # Enables buffer with top down map of the current episode/level.
    game.set_automap_buffer_enabled(True)
    game.set_automap_mode(vzd.AutomapMode.OBJECTS_WITH_SIZE)

    game.add_available_game_variable(vzd.GameVariable.POSITION_X)
    game.add_available_game_variable(vzd.GameVariable.POSITION_Y)
    # game.add_available_game_variable(vzd.GameVariable.POSITION_Z)
    game.add_available_game_variable(vzd.GameVariable.ANGLE)

    # Enables information about all objects present in the current episode/level.
    game.set_objects_info_enabled(True)

    # Enables information about all sectors (map layout).
    game.set_sectors_info_enabled(True)

    # Sets other rendering options (all of these options except crosshair are enabled (set to True) by default)
    game.set_render_hud(False)
    game.set_render_minimal_hud(False)  # If hud is enabled
    game.set_render_crosshair(False)
    game.set_render_weapon(True)
    game.set_render_decals(False)  # Bullet holes and blood on the walls
    game.set_render_particles(False)
    game.set_render_effects_sprites(False)  # Smoke and blood
    game.set_render_messages(False)  # In-game messages
    game.set_render_corpses(False)
    game.set_render_screen_flashes(True)  # Effect upon taking damage or picking up items

    # Adds buttons that will be allowed.
    # game.add_available_button(vzd.Button.MOVE_FORWARD_BACKWARD_DELTA)
    game.add_available_button(vzd.Button.MOVE_FORWARD)
    game.add_available_button(vzd.Button.TURN_LEFT_RIGHT_DELTA)
    # game.add_available_button(vzd.Button.TURN_LEFT_RIGHT_DELTA)
    # game.add_available_button(vzd.Button.ATTACK)

    # Adds game variables that will be included in state. And ammo is only a prop for the doomgame
    # game.add_available_game_variable(vzd.GameVariable.AMMO0)

    # Causes episodes to finish after 200 tics (actions)
    game.set_episode_timeout(200)

    # Makes episodes start after 10 tics (~after raising the weapon)
    game.set_episode_start_time(10)

    # Makes the window appear (turned on by default), but here turn it off to avoid memory leak (a strange problem)
    game.set_window_visible(False)

    # Turns on the sound. (turned off by default)
    game.set_sound_enabled(True)

    # Sets the livin reward (for each move) to -1
    game.set_living_reward(-1)

    # Sets ViZDoom mode (PLAYER, ASYNC_PLAYER, SPECTATOR, ASYNC_SPECTATOR, PLAYER mode is default)
    game.set_mode(vzd.Mode.PLAYER)

    # Enables engine output to console.
    #game.set_console_enabled(True)

    # Initialize the game. Further configuration won't take any effect from now on.
    game.init()

    # Define some actions. Each list entry corresponds to declared buttons:
    # MOVE_LEFT, MOVE_RIGHT, ATTACK
    # game.get_available_buttons_size() can be used to check the number of available buttons.
    # 5 more combinations are naturally possible but only 3 are included for transparency when watching.
    actions = [[True, 0], [False, 90], [False, -90]] # this is to set the specific value for the
    # action the agent chooses. Meanwhile, the button need to choose the *_DELTA.

    # Run this many episodes
    episodes = 1000000

    # Sets time that will pause the engine after each action (in seconds)
    # Without this everything would go too fast for you to keep track of what's happening.
    sleep_time = 1.0 / vzd.DEFAULT_TICRATE  # = 0.028
    # sleep_time = 28
    for i in range(episodes):
        index = i%100
        game.set_doom_map("map{}".format(index))
        # this is for showing the map in a list
        MAP = []
        with open('../../../NavDoom/outputs/sources/7_TRAIN_MAP{:02d}.txt'.format(index), 'r') as f:
            for line in f:
                A_line = list(line.strip('\n'))
                A_line = [float(x) if type(x) is str else None for x in A_line]
                MAP.append(A_line)
        map_array = np.array(MAP)
        print("Episode #" + str(i + 1))

        state = game.get_state()
        n = state.number
        vars = state.game_variables
        map = state.automap_buffer
        if map is not None:
            plt.imshow(map, cmap='gray')
            plt.show()

        # Starts a new episode. It is not needed right after init() but it doesn't cost much. At least the loop is nicer.
        game.new_episode()
        step = 0

        # while not game.is_episode_finished():
        while step < 5:
            step += 1

            # Which consists of:
            # n = state.number

            # screen_buf = state.screen_buffer
            # depth_buf = state.depth_buffer
            # labels_buf = state.labels_buffer
            # automap_buf = state.automap_buffer
            # labels = state.labels
            # objects = state.objects
            # sectors = state.sectors

            # Games variables can be also accessed via:
            #game.get_game_variable(GameVariable.AMMO2)

            # Makes a random action and get remember reward.
            action = choice(actions)
            r = game.make_action(action)

            # Gets the state
            state = game.get_state()
            n = state.number
            vars = state.game_variables
            # The same could be achieved with:
            # game.set_action(choice(actions))
            # game.advance_action(skiprate)
            # r = game.get_last_reward()

            # Prints state's game variables and reward.
            print("State #" + str(n))
            print("Game variables:", vars)
            print( "Action:", action)
            print("=====================")

            # Print out its position
            Position_X = (vars[0] - 48) / 96
            Position_Y = (vars[1] - 48) / 96
            position_map = copy.deepcopy(map_array)
            position_map[int(Position_Y)][int(Position_X)] = '2'
            print(position_map)

            # show the map
            map = state.automap_buffer
            if map is not None:
                plt.imshow(map, cmap='gray')
                plt.show()
            if sleep_time > 0:
                sleep(sleep_time)

        # Check how the episode went.
        print("Episode finished.")
        print("Total reward:", game.get_total_reward())
        print("************************")

    # It will be done automatically anyway but sometimes you need to do it in the middle of the program...
    game.close()
