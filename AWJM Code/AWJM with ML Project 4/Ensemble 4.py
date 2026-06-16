import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def create_awjm_architecture_diagram():
    fig, ax = plt.subplots(figsize=(18, 10))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 11)
    ax.axis('off')

    plt.title('AWJM Ensemble Neural Network Architecture',
              fontsize=18, fontweight='bold', pad=10)

    # ===== Colors =====
    input_color    = '#FFDDC1'
    model_colors   = ['#C1FFD7', '#C1E0FF', '#FFC1E0', '#FFFEC1', '#E0C1FF', '#C1FFFF']
    ensemble_color = '#D3D3D3'  #
    output_color   = '#D1C1FF'
    line_color     = '#333333'

    # Box sizes
    box_w, box_h = 3.2, 1.0

    # ===== Helper: Draw Box =====
    def draw_box(x, y, text, fc, fontsize=10):
        ax.add_patch(Rectangle((x, y), box_w, box_h,
                               facecolor=fc, edgecolor='black', lw=1.2))
        ax.text(x + box_w/2, y + box_h/2, text,
                ha='center', va='center',
                fontsize=fontsize, fontweight='bold',  # **Bold text**
                wrap=True)

    # ==== Input Layer ====
    ax.text(2.5, 10, 'Input Layer', fontsize=14, fontweight='bold', ha='center')

    input_params = [
        'Abrasive Pressure\n(MPa)',
        'Standoff Distance\n(mm)',
        'Traverse Speed\n(mm/min)',
        'Mass Flow Rate\n(kg/min)'
    ]

    start_y_in = 8.0
    gap_in = 1.0
    y_positions_in = [start_y_in - i * (box_h + gap_in) for i in range(len(input_params))]

    for y, param in zip(y_positions_in, input_params):
        draw_box(0.8, y, param, input_color)

    # ==== Individual Models ====
    ax.text(7.0, 10, 'Individual Models', fontsize=14, fontweight='bold', ha='center')

    model_boxes = [
        ('Ridge Regression', 'Linear model\nL2 Regularization'),
        ('Random Forest', '200 Trees\nBootstrap=True'),
        ('XGBoost', '300 Estimators\nlr=0.05, max_depth=6'),
        ('Neural Network', '3×ReLU Layers\nAdam, Epochs=200'),
        ('SVR', 'RBF Kernel\nC, γ tuned'),
        ('ElasticNet', 'L1 + L2 Regularization\nα, l1_ratio tuned')
    ]

    start_y_model = 8.8
    gap_model = 0.6
    y_positions_model = [start_y_model - i * (box_h + gap_model) for i in range(len(model_boxes))]

    for (title, desc), y, c in zip(model_boxes, y_positions_model, model_colors):
        draw_box(5.5, y, f"{title}\n{desc}", c, fontsize=9)

    # ==== Ensemble Averaging ====
    ax.text(11.5, 10, 'Ensemble Averaging', fontsize=14, fontweight='bold', ha='center')

    ensemble_y = 4.6
    orig_w, orig_h = box_w, box_h
    box_w, box_h = 3.2, 1.5  # slightly larger box for ensemble

    draw_box(
        10.0, ensemble_y,
        'Weighted Average \nRidge(0.10), RandomForest(0.15)\nSVR(0.20),ElasticNet(0.15)\nXGBoost(0.20), NN(0.20)',
        ensemble_color,
        fontsize=10
    )

    box_w, box_h = orig_w, orig_h  # restore original size

    # ==== Output Layer ====
    ax.text(15.8, 10, 'Output Layer', fontsize=14, fontweight='bold', ha='center')

    out_offsets = [1.0, -1.0]  # Removed Kerf
    outputs = [
        'Surface Roughness\n(SR, µm)',
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

    legend_elements = [
        Rectangle((0, 0), 1, 1, fc=input_color, label='Input Layer'),
        Rectangle((0, 0), 1, 1, fc='#C1FFD7', label='Individual Models'),
        Rectangle((0, 0), 1, 1, fc=ensemble_color, label='Ensemble Averaging'),  # light gray
        Rectangle((0, 0), 1, 1, fc=output_color, label='Output Layer')

    ]

    ax.legend(handles=legend_elements, loc='upper center',
              bbox_to_anchor=(0.5, -0.03), ncol=4, fontsize=12, frameon=False)

    # ==== Process Flow ====
    ax.text(9, 0.3,
            'Process Flow: Input → Individual Models → Ensemble Averaging → Output',
            fontsize=15, style='italic', fontweight='bold', ha='center')

    plt.tight_layout()
    plt.show()


# Run
create_awjm_architecture_diagram()
