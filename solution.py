#!/usr/bin/env python

import gym
import gym_duckietown_agent  # DO NOT CHANGE THIS IMPORT (the environments are defined here)
from duckietown_challenges import wrap_solution, ChallengeSolution, ChallengeInterfaceSolution


def solve(params, cis):
    # python has dynamic typing, the line below can help IDEs with autocompletion
    assert isinstance(cis, ChallengeInterfaceSolution)
    # after this cis. will provide you with some autocompletion in some IDEs (e.g.: pycharm)
    cis.info('Creating model.')
    # you can have logging capabilties through the solution interface (cis).
    # the info you log can be retrieved from your submission files.

    # BEGIN SUBMISSION
    # if you have a model class with a predict function this are likely the only lines you will need to modifiy
    from model import TfInference
    # define observation and output shapes
    model = TfInference(observation_shape=(1, 120, 160, 3),  # this is the shape of the image we get.
                        action_shape=(1, 2),  # we need to output v, omega.
                        graph_location='tf_models/')  # this is the folder where our models are stored.
    # END SUBMISSION

    # We get environment from the Evaluation Engine
    cis.info('Making environment')
    env = gym.make(params['env'])
    # Then we make sure we have a connection with the environment and it is ready to go
    cis.info('Reset environment')
    observation = env.reset()
    # While there are no signal of completion (simulation done)
    # we run the predictions for a number of episodes, don't worry, we have the control on this part
    while True:
        # we passe the observation to our model, and we get an action in return
        action = model.predict(observation)
        # we tell the environment to perform this action and we get some info back in OpenAI Gym style
        observation, reward, done, info = env.step(action)
        # here you may want to compute some stats, like how much reward are you getting
        # notice, this reward may no be associated with the challenge score.

        # it is important to check for this flag, the Evalution Engine will let us know when should we finish
        # if we are not careful with this the Evaluation Engine will kill our container and we will get no score
        # from this submission
        if 'simulation_done' in info:
            break
        if done:
            env.reset()

    # release CPU/GPU resources, let's be friendly with other users that may need them
    model.close()


class Submission(ChallengeSolution):
    def run(self, cis):
        assert isinstance(cis, ChallengeInterfaceSolution)  # this is a hack that would help with autocompletion

        # get the configuration parameters for this challenge
        params = cis.get_challenge_parameters()
        cis.info('Parameters: %s' % params)

        output = {'status': 'success'}
        try:
            cis.info('Starting.')
            solve(params, cis)  # let's try to solve the challenge, exciting ah?
        except Exception as e:
            output['status'] = 'failure'
            output['msg'] = str(e)

        cis.set_solution_output_dict(output)
        cis.info('Finished.')


if __name__ == '__main__':
    print('Starting submission')
    wrap_solution(Submission())