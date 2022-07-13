# Experiment class

class Experiment:


  def __init__(self, run_name, experiment_id, nn_hyps):
    
    self.run_name = run_name
    self.experiment_id = experiment_id
    self.nn_hyps = nn_hyps

    self.results = None
    self.is_trained = False

  def train(self):
    pass

  def load_results(self):
    pass

  def __str__(self):
    return f'''
    Run Name: {self.run_name}
    Experiment ID: {self.experiment_id}
    nn_hyps: {self.nn_hyps}
    is_trained: {self.is_trained}
    '''


  
