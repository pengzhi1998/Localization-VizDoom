import numpy as np
import numpy.random as npr
from vizdoom import *

def set_doom_configurations(game, args):
    game.set_doom_scenario_path(args.path)

    # Sets map to start (scenario .wad files can contain many maps). this step is set after initialization
    # game.set_doom_map(args.map_number)

    # Sets resolution. Default is 320X240.
    game.set_screen_resolution(ScreenResolution.RES_640X480)

    # Sets the screen buffer format. Not used here but now you can change it. Default is CRCGCB.
    game.set_screen_format(ScreenFormat.RGB24)

    # Enables depth buffer.
    game.set_depth_buffer_enabled(True)

    # Enables labeling of in game objects labeling.
    game.set_labels_buffer_enabled(True)

    # Enables buffer with top down map of the current episode/level.
    game.set_automap_buffer_enabled(True)
    game.set_automap_mode(AutomapMode.OBJECTS_WITH_SIZE)

    game.add_available_game_variable(GameVariable.POSITION_X)
    game.add_available_game_variable(GameVariable.POSITION_Y)
    game.add_available_game_variable(GameVariable.ANGLE)

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
    game.add_available_button(Button.MOVE_FORWARD)
    game.add_available_button(Button.TURN_LEFT_RIGHT_DELTA)
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
    game.set_mode(Mode.PLAYER)

    return game


def get_depth(map_design, position, orientation):
    m, n = map_design.shape
    depth = 0
    new_tuple = position
    while(compare_tuples(new_tuple, tuple([m - 1, n - 1])) and
            compare_tuples(tuple([0, 0]), new_tuple)):
        if map_design[new_tuple] != 0:
            break
        else:
            new_tuple = get_tuple(new_tuple, orientation)
            depth += 1
    return depth


def get_next_state(map_design, position, orientation, action):
    m, n = map_design.shape
    if action == 'TURN_LEFT' or action == 0:
        orientation = (orientation + 1) % 4
    elif action == "TURN_RIGHT" or action == 1:
        orientation = (orientation - 1) % 4
    elif action == "MOVE_FORWARD" or action == 2:
        new_tuple = get_tuple(position, orientation)
        if compare_tuples(new_tuple, tuple([m - 1, n - 1])) and \
           compare_tuples(tuple([0, 0]), new_tuple) and \
           map_design[new_tuple] == 0:
            position = new_tuple
    return position, orientation


def get_random_position(map_design):
    m, n = map_design.shape
    while True:
        index = tuple([np.random.randint(m), np.random.randint(n)])
        if map_design[index] == 0:
            return index


def get_tuple(i, orientation):
    if orientation == 0 or orientation == "east":
        new_tuple = tuple([i[0], i[1] + 1])
    elif orientation == 2 or orientation == "west":
        new_tuple = tuple([i[0], i[1] - 1])
    elif orientation == 1 or orientation == "north":
        new_tuple = tuple([i[0] - 1, i[1]])
    elif orientation == 3 or orientation == "south":
        new_tuple = tuple([i[0] + 1, i[1]])
    else:
        assert False, "Invalid orientation"
    return new_tuple


def compare_tuples(a, b):
    """
    Returns true if all elements of a are less than
    or equal to b
    """
    assert len(a) == len(b), "Unequal lengths of tuples for comparison"
    for i in range(len(a)):
        if a[i] > b[i]:
            return False
    return True


if __name__ == '__main__':
    print(generate_map(7))
