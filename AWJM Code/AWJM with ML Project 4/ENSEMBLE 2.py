import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def create_awjm_architecture_diagram():
    fig, ax = plt.subplots(figsize=(18, 10))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 11)
    ax.axis('off')

    plt.title('AWJM Ensemble Neural Network Architecture',
              fontsize=22, fontweight='bold', pad=20)

    # ===== Colors =====
    input_color = '#FFDDC1'  # input boxes
    model_colors = ['#C1FFD7', '#C1E0FF', '#FFC1E0', '#FFFEC1']  # models
    ensemble_color = '#D3D3D3'  # gray color for Weighted Average
    output_color = '#D1C1FF'  # output boxes
    line_color = '#333333'

    # ===== Box sizes =====
    box_w, box_h = 3.5, 1.5  # increased height for better text fit

    # ===== Helper: Draw Box =====
    def draw_box(x, y, text, fc, fontsize=12, bold=True):
        ax.add_patch(Rectangle((x - box_w / 2, y - box_h / 2), box_w, box_h,
                               facecolor=fc, edgecolor='black', lw=1.4))
        ax.text(x, y, text, ha='center', va='center',
                fontsize=fontsize, fontweight='bold' if bold else 'normal',
                wrap=True)

    # ===== Layer X positions =====
    x_input = 3.0
    x_model = 8.0
    x_ens   = 12.0
    x_out   = 16.0

    # ==== Input Layer ====
    ax.text(x_input, 10.3, 'Input Layer', fontsize=16, fontweight='bold', ha='center')

    input_params = [
        'Abrasive Pressure\n(MPa)',
        'Standoff Distance\n(mm)',
        'Traverse Speed\n(mm/min)',
        'Mass Flow Rate\n(kg/min)'
    ]
    y_inputs = [8.5, 6.5, 4.5, 2.5]  # increased vertical spacing

    for y, param in zip(y_inputs, input_params):
        draw_box(x_input, y, param, input_color, fontsize=12)

    # ==== Individual Models ====
    ax.text(x_model, 10.3, 'Individual Models', fontsize=16, fontweight='bold', ha='center')

    model_boxes = [
        ('Ridge Regression',
         'Linear model\nL2 regularization\nα tuned by CV'),
        ('Random Forest',
         'n_estimators=200\nmax_depth=None\nbootstrap=True'),
        ('ElasticNet',
         'Linear model\nL1 & L2 regularization\nα & l1_ratio tuned'),
        ('SVR',
         'Kernel=RBF\nC, gamma tuned\nEpsilon insensitive')
    ]
    y_models = [8.5, 6.5, 4.5, 2.5]  # aligned with input boxes

    for (title, desc), y, c in zip(model_boxes, y_models, model_colors):
        draw_box(x_model, y, f"{title}\n{desc}", c, fontsize=11)

    # ==== Ensemble Averaging ====
    ax.text(x_ens, 10.3, 'Ensemble Averaging', fontsize=16, fontweight='bold', ha='center')

    draw_box(x_ens, 5.0,
             'Weighted Average\nRidge(0.15), RF(0.25)\nElasticNet(0.30), SVR(0.30)',
             ensemble_color, fontsize=12)

    # ==== Output Layer ====
    ax.text(x_out, 10.3, 'Output Layer', fontsize=16, fontweight='bold', ha='center')

    outputs = [
        ('Surface Roughness\n(SR, µm)', 6.0),
        (  'Kerf Angle\n(KA°)', 4.0)
    ]
    for label, y in outputs:
        draw_box(x_out, y, label, output_color, fontsize=12)

    # ==== Arrows ====
    arrowprops = dict(arrowstyle='->', color=line_color, lw=1.8)

    # Input → Models
    for y_in in y_inputs:
        for y_m in y_models:
            ax.annotate('', xy=(x_model - box_w/2, y_m),
                        xytext=(x_input + box_w/2, y_in),
                        arrowprops=arrowprops)

    # Models → Ensemble
    for y_m in y_models:
        ax.annotate('', xy=(x_ens - box_w/2, 5.0),
                    xytext=(x_model + box_w/2, y_m),
                    arrowprops=arrowprops)

    # Ensemble → Outputs
    for _, y_out in outputs:
        ax.annotate('', xy=(x_out - box_w/2, y_out),
                    xytext=(x_ens + box_w/2, 5.0),
                    arrowprops=arrowprops)

    legend_elements = [
        Rectangle((0, 0), 1, 1, fc=input_color, label='Input Layer'),
        Rectangle((0, 0), 1, 1, fc='#C1FFD7', label='Individual Models'),
        Rectangle((0,0), 1, 1, fc=ensemble_color, label='Ensemble Averaging'),  # light gray
        Rectangle((0, 0), 1, 1, fc=output_color, label='Output Layer')


    ]

    ax.legend(handles=legend_elements, loc='upper center',
              bbox_to_anchor=(0.5, -0.03), ncol=4, fontsize=12 , frameon=False)

    # ==== Process Flow ====
    ax.text(9, 0.5,
            'Process Flow: Input → Individual Models → Ensemble Averaging → Output',
            fontsize=15, style='italic', fontweight='bold', ha='center')

    plt.tight_layout()
    plt.show()


# Run
create_awjm_architecture_diagram()
