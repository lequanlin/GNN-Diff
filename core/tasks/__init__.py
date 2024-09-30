from .node_classification import NCFTask
from .node_classification_large import NCFTask_Large
from .node_classification_lr import NCFTask_LR
from .link_prediction import LPTask

tasks = {
    'node_classification': NCFTask,
    'node_classification_large': NCFTask_Large,
    'node_classification_lr': NCFTask_LR,
    'link_prediction': LPTask,
}