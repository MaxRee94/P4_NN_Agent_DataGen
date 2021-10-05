from tensorflow import keras
import random


def get_trials():
    trials = []
    for g in range(1, 25):
        blueprint = [
            {
                "width": 77,
                "activation": None,
            },
            {
                "width": 128,
                "activation": "relu"
            },
            {
                "width": 3,
                "activation": "relu"
            }
        ]
        for i in range(30):
            # Change hidden layer widths
            _blueprint = []
            for j, layer in enumerate(blueprint):
                if j == 0 or j == (len(blueprint) - 1):
                    _blueprint.append(layer.copy())
                else:
                    hidden_layer = layer.copy()
                    hidden_layer["width"] = ((i % 6) + 1) * 128
                    _blueprint.append(hidden_layer)
            blueprint = _blueprint

            # Add a hidden layer
            if i % 6 == 0 and i != 0:
                blueprint.insert(
                    -1,
                    hidden_layer.copy()
                )

            trial = {
                "compilation_settings": {
                    "optimizer": keras.optimizers.Adam(learning_rate=0.003),
                    "loss": "categorical_crossentropy",
                    "metrics": ["accuracy"],
                },
                "blueprint": blueprint,
                "training_settings": {"epochs": 4 * g},
                "learnrate": 0.003,
            }
            trials.append(trial)

    return trials

learnrate = 0.001
trials = get_trials()
trials = [
    {'compilation_settings': {
        'optimizer': keras.optimizers.Adam(learning_rate=learnrate),
        'loss': 'categorical_crossentropy',
        'metrics': ['accuracy']
    },
    'blueprint': [
        {'width': 75, 'activation': None}, {'width': 200, 'activation': 'relu'}, {'width': 50, 'activation': 'relu'}, {'width': 3, 'activation': 'relu'}
    ],
    'training_settings': {'epochs': 50, "batch_size": 32},
    'learnrate': None
    }
]
