import json
import matplotlib.pyplot as plt
import numpy as np
import os

# Configure matplotlib style
plt.style.use('ggplot')

def load_data(file_path):
    data = []
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return []
    
    with open(file_path, 'r') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except:
                pass
    return data

def plot_component_contribution(data):
    """
    Generates a stacked bar chart showing how different components contribute to the score.
    """
    # Filter for valid data
    samples = [d for d in data if 'component_scores' in d]
    if not samples:
        print("No sample data found for component analysis.")
        return

    # Take last 5 samples for clarity
    samples = samples[-5:]
    
    labels = [f"Sample {i+1}" for i in range(len(samples))]
    
    # Extract scores (normalized to 0-10 scale for visualization)
    keywords = [d['component_scores'].get('keyword_pct', 0) / 10 for d in samples]
    semantic = [d['component_scores'].get('semantic_pct', 0) / 10 for d in samples]
    grammar = [d['component_scores'].get('grammar_pct', 0) / 10 for d in samples]
    presentation = [d['component_scores'].get('presentation_pct', 0) / 10 for d in samples]
    
    x = np.arange(len(labels))
    width = 0.6

    fig, ax = plt.subplots(figsize=(10, 6))

    # Stacked bars
    p1 = ax.bar(x, keywords, width, label='Keyword Match')
    p2 = ax.bar(x, semantic, width, bottom=keywords, label='Semantic Similarity')
    # p3 = ax.bar(x, grammar, width, bottom=np.array(keywords)+np.array(semantic), label='Grammar') # Grammar is often 0, skipping to avoid clutter if empty
    
    ax.set_ylabel('Score Contribution (Scaled)')
    ax.set_title('Pipeline Component Contribution to Final Grade')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    plt.tight_layout()
    plt.savefig('analysis_component_contribution.png')
    print("Saved analysis_component_contribution.png")

def plot_ai_vs_human(data):
    """
    Generates a grouped bar chart comparing AI Grades vs Human Grades.
    """
    # Filter for records with human_marks
    samples = [d for d in data if 'human_marks' in d and 'llm_output' in d]
    
    if not samples:
        # If no human marks found, use dummy data for demonstration (Since user asked "how can this analysis be done")
        print("No human_marks found in data. Using dummy data for demonstration.")
        labels = ['Sample A', 'Sample B', 'Sample C', 'Sample D']
        human_scores = [8.5, 9.0, 7.5, 6.0] # Dummy
        ai_scores = [8.2, 8.8, 7.0, 6.5]   # Dummy
    else:
        labels = [f"S{i+1}" for i in range(len(samples))]
        human_scores = [d['human_marks'] for d in samples]
        # Use LLM recommended marks, fallback to deterministic
        ai_scores = []
        for d in samples:
            llm_marks = d.get('llm_output', {}).get('recommended_marks', 0)
            det_marks = d.get('deterministic_recommended_marks', 0)
            ai_scores.append(llm_marks if llm_marks > 0 else det_marks)

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, human_scores, width, label='Human Grade')
    rects2 = ax.bar(x + width/2, ai_scores, width, label='AI System Grade')

    ax.set_ylabel('Grade (0-10)')
    ax.set_title('Performance Comparison: Human vs AI Grader')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    # Add labels on top
    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)

    plt.tight_layout()
    plt.savefig('analysis_ai_vs_human.png')
    print("Saved analysis_ai_vs_human.png")

if __name__ == "__main__":
    file_path = "eval_records.jsonl" # Adjust path if needed
    data = load_data(file_path)
    
    if data:
        plot_component_contribution(data)
        plot_ai_vs_human(data)
    else:
        print("No data found to plot.")
