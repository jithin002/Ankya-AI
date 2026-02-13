import sys
import os
import json

if len(sys.argv) < 3:
    print("Usage: python eval_with_model.py <model_dir> <image_path>")
    sys.exit(1)

model_dir = sys.argv[1]
image = sys.argv[2]

# ensure project root importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pipelines import icr_pipeline3 as p

print('Initializing TrOCR with', model_dir)
p.trocr_init(model_name=model_dir)

reference = {
    "max_marks": 10,
    "keywords": [
        {"term": "Newton's first law", "weight": 1.0},
        {"term": "inertia", "weight": 1.0},
        {"term": "Newton's second law", "weight": 1.0},
        {"term": "F=ma", "weight": 1.0},
        {"term": "Newton's third law", "weight": 1.0}
    ],
    "reference_long": (
        "Newton's laws of motion: "
        "1) (Inertia) A body remains at rest or in uniform motion unless acted on by a net external force. "
        "2) (F=ma) The net force acting on a body equals mass times acceleration. "
        "3) (Action-Reaction) For every action there is an equal and opposite reaction."
    ),
    "reference_short": [
        "inertia",
        "F = m a",
        "action reaction",
        "net force = mass * acceleration"
    ]
}

res = p.evaluate_from_image(image, "Explain Newton's three laws of motion with examples.", reference, debug_save_image=True)
print(json.dumps(res, indent=2))
