import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

def create_awjm_architecture_diagram():
    fig, ax = plt.subplots(figsize=(18, 10))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 10)
    ax.axis('off')

    plt.title('AWJM Ensemble Neural Network Architecture',
              fontsize=18, fontweight='bold', pad=5)

    # Colors
    input_color    = '#FFDDC1'
    model_colors   = ['#C1FFD7', '#C1E0FF', '#FFC1E0', '#FFFEC1']
    ensemble_color = '#FFD8C1'
    output_color   = '#D1C1FF'
    line_color     = '#444444'

    box_w, box_h = 3.6, 1.2  # Standard box size

    # Helper to draw box with centered text
    def draw_box(x, y, text, fc, fontsize=11, bold=False):
        ax.add_patch(Rectangle((x, y), box_w, box_h,
                               facecolor=fc, edgecolor='black', lw=1.2))
        ax.text(x + box_w/2, y + box_h/2, text,
                ha='center', va='center',
                fontsize=fontsize,
                fontweight='bold' if bold else 'normal',
                wrap=True)

    # ==== Input Layer ====
    ax.text(2.5, 9, 'Input Layer', fontsize=14, fontweight='bold', ha='center')
    input_params = [
        'Abrasive Pressure\n(MPa)',
        'Standoff Distance\n(mm)',
        'Traverse Speed\n(mm/min)',
        'Mass Flow Rate\n(kg/min)'
    ]
    y_positions_in = np.linspace(7.0, 2.6, len(input_params))
    for y, param in zip(y_positions_in, input_params):
        draw_box(0.8, y, param, input_color)

    # ==== Individual Models ====
    ax.text(7.0, 9, 'Individual Models', fontsize=14, fontweight='bold', ha='center')
    model_boxes = [
        ('Ridge Regression', 'Linear model\nL2 regularization\nα tuned by CV'),
        ('Random Forest', 'n_estimators=200\nmax_depth=None\nbootstrap=True'),
        ('XGBoost', 'n_estimators=300\nlearning_rate=0.05\nmax_depth=6'),
        ('Neural Network', 'Hidden1: (ReLU)\nHidden2: (ReLU)\nHidden3: (ReLU)\nAdam (lr=0.001)\nEpochs=200')
    ]
    y_positions_model = np.linspace(7.0, 2.6, len(model_boxes))
    for (title, desc), y, c in zip(model_boxes, y_positions_model, model_colors):
        # Decreased fontsize to 9 and removed bold
        draw_box(5.5, y, f"{title}\n{desc}", c, fontsize=9, bold=False)

    # ==== Ensemble ====
    ax.text(12.0, 9, 'Ensemble Averaging', fontsize=14, fontweight='bold', ha='center')
    ensemble_y = 4.8
    draw_box(10.0, ensemble_y,
             'Weighted Average\nRidge(0.15), RF(0.25)\nXGBoost(0.30), NN(0.30)',
             ensemble_color, bold=True)

    # ==== Output Layer ====
    ax.text(15.8, 9, 'Output Layer', fontsize=14, fontweight='bold', ha='center')

    # Three outputs: SR, MRR, Kerf (vertical spacing)
    out_offsets = [2.0, 0.0, -2.0]  # positions relative to ensemble_y
    outputs = [
        'Surface Roughness\n(SR, µm)',
        'Material Removal Rate\n(MRR, mm³/min)',
        'Kerf Angle\n(KA°)'
    ]
    for off, label in zip(out_offsets, outputs):
        draw_box(14.2, ensemble_y + off, label, output_color)

    # ==== Arrows ====
    arrowprops = dict(arrowstyle='->', color=line_color, lw=1.5)

    # Input → Models
    for y_in in y_positions_in:
        for y_m in y_positions_model:
            ax.annotate('', xy=(5.5, y_m + box_h/2),
                        xytext=(0.8 + box_w, y_in + box_h/2),
                        arrowprops=arrowprops)

    # Models → Ensemble
    for y_m in y_positions_model:
        ax.annotate('', xy=(10.0, ensemble_y + box_h/2),
                    xytext=(5.5 + box_w, y_m + box_h/2),
                    arrowprops=arrowprops)

    # Ensemble → Outputs
    for off in out_offsets:
        ax.annotate('', xy=(14.2, ensemble_y + off + box_h/2),
                    xytext=(10.0 + box_w, ensemble_y + box_h/2),
                    arrowprops=arrowprops)

    # Process Flow
    ax.text(9, 1,
            'Process Flow: Input → Models → Ensemble Averaging → Output',
            fontsize=25, style='italic', ha='center')

    plt.tight_layout()
    plt.show()

# Run the function
create_awjm_architecture_diagram()
