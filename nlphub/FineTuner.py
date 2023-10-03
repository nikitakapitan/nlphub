from nlphub import Trainer
import logging
import evaluate

class FineTuner(Trainer):

    def __init__(self, config):
        super().__init__(config)
        self.load_metrics()         # self.metrics

    def load_metrics(self):
        """ Output example:
        {
            'accuracy': {
                'func': evaluate.accuracy,
                'args': None
            },
            'f1': {
                'func': evaluate.f1,
                'args': {'average': 'weighted'}
            }
        }                      
        """
        metrics_functions = {}
        for metric_config in self.config['METRIC_NAMES']:

            metrics_functions[metric_config['name']] = {
                'func': evaluate.load(metric_config['name']),
                'args': metric_config.get('args', {})}
            
        self.metrics_functions = metrics_functions


    def define_compute_metrics(self):

        def compute_metrics(eval_pred):
            """ Output example:
            {
                'accuracy'  : 0.9518,
                'f1'        : 0.8756,
            }                      
            """
            logits, labels = eval_pred
            preds = logits.argmax(axis=-1)

            metrics = {}
            for metric_name, metric_data  in self.metrics_functions.items():
                metric_func = metric_data['func']
                metric_args = metric_data['args']
                metrics[metric_name] = metric_func.compute(predictions=preds, references=labels, **metric_args)
                
            return metrics
        
        self.compute_metrics_func = compute_metrics # function
