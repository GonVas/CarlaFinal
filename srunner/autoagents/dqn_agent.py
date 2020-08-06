from __future__ import print_function

import carla

from srunner.autoagents.rl_agent import RlAgent

from algos import DQN_carla


class DqnAgent(RlAgent):


    def setup(self, path_to_conf_file):
        """
        Initialize everything needed by your agent and set the track attribute to the right type:
        """

        self.current_control = carla.VehicleControl()
        self.current_control.steer = 0.0
        self.current_control.throttle = 1.0
        self.current_control.brake = 0.0
        self.current_control.hand_brake = False

        
        
        pass


    def run_step(self, input_data, timestamp):
	    """
	    Execute one step of navigation.
	    :return: control
	    """
	    control = carla.VehicleControl()
	    control.steer = 0.0
	    control.throttle = 0.0
	    control.brake = 0.0
	    control.hand_brake = False

	    print(input_data)

	    return control