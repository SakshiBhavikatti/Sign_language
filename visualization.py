import os
import matplotlib.pyplot as plt
from PIL import Image

predictions = [
    {"confidence": 0.9973, "filename": "A_17.png", "predicted_class": "A"},
    {"confidence": 0.9998, "filename": "A_3.png", "predicted_class": "A"},
    {"confidence": 0.9751, "filename": "D_4.png", "predicted_class": "D"},
    {"confidence": 0.9611, "filename": "D_8.png", "predicted_class": "D"},
    {"confidence": 0.9998, "filename": "E_15.png", "predicted_class": "E"},
    {"confidence": 0.9995, "filename": "F_1.png", "predicted_class": "F"},
    {"confidence": 0.9985, "filename": "G_0.png", "predicted_class": "G"},
    {"confidence": 0.8571, "filename": "H_14.png", "predicted_class": "H"},
    {"confidence": 0.9997, "filename": "H_18.png", "predicted_class": "H"},
    {"confidence": 0.9872, "filename": "H_19.png", "predicted_class": "H"},
    {"confidence": 0.9448, "filename": "H_9.png", "predicted_class": "H"},
    {"confidence": 1.0, "filename": "I_10.png", "predicted_class": "I"},
    {"confidence": 1.0, "filename": "I_11.png", "predicted_class": "I"},
    {"confidence": 1.0, "filename": "J_2.png", "predicted_class": "K"},
    {"confidence": 0.9887, "filename": "J_6.png", "predicted_class": "K"},
    {"confidence": 0.9953, "filename": "L_13.png", "predicted_class": "M"},
    {"confidence": 1.0, "filename": "N_7.png", "predicted_class": "O"},
    {"confidence": 0.9675, "filename": "U_12.png", "predicted_class": "V"},
    {"confidence": 0.9999, "filename": "U_5.png", "predicted_class": "V"},
    {"confidence": 0.9957, "filename": "V_16.png", "predicted_class": "W"}
]

image_dir = 'test_images'

cols = 5
rows = (len(predictions) + cols - 1) // cols
plt.figure(figsize=(16, 3.5 * rows))

for idx, pred in enumerate(predictions):
    filepath = os.path.join(image_dir, pred['filename'])
    true_label = pred['filename'].split('_')[0]
    predicted_label = pred['predicted_class']
    confidence = pred['confidence']
    
    image = Image.open(filepath)
    plt.subplot(rows, cols, idx + 1)
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    
    title = f"True: {true_label}\nPred: {predicted_label} ({confidence:.2%})"
    color = 'green' if true_label == predicted_label else 'red'
    plt.title(title, color=color, fontsize=10)

plt.tight_layout()
plt.suptitle("Prediction Results on Test Images", fontsize=18, y=1.02)
# plt.savefig("prediction_report.png", bbox_inches='tight')
plt.show()
