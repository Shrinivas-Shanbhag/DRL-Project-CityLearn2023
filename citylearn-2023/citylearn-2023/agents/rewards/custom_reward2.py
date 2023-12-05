from citylearn.reward_function import SolarPenaltyAndComfortReward

class ComfortRewardFunction2(SolarPenaltyAndComfortReward):
    """ Simple passthrough example of comfort reward from Citylearn env """
    def __init__(self, env_metadata):
        super().__init__(env_metadata)
        self.env = None

    def calculate(self, observations):
        citylearn_metrics = self.env.evaluate_citylearn_challenge()
        score_dict = citylearn_metrics['average_score']
        value = score_dict.get('value', 0.0)

        return [float(value)]
