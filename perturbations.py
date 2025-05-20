from PIL import ImageEnhance
import numpy as np
import cv2
from PIL import Image

from metrics import evaluate_model

def apply_transformations(img_pil, transform_name):
    """
    Function applies a various transformations to PIL image:
        - brightness up and down,
        - contrast up and down,
        - saturation up and down,
        - rotate 15 and -15 degrees,
        - gaussian blur,
        - color shift.
    Args:
        img_pil (PIL Image): PIL Image to be transformed.
        transform_name (str): Name of transformation to apply.
    Returns:
        img_pil (PIL Image): PIL Image after transformation.
    """

    if isinstance(img_pil, np.ndarray):
        img_pil = Image.fromarray(img_pil)

    if transform_name == "brightness_down":
        img_pil = ImageEnhance.Brightness(img_pil).enhance(0.5)
    elif transform_name == "brightness_up":
        img_pil = ImageEnhance.Brightness(img_pil).enhance(1.5)
    elif transform_name == "contrast_down":
        img_pil = ImageEnhance.Contrast(img_pil).enhance(0.5)
    elif transform_name == "contrast_up":
        img_pil = ImageEnhance.Contrast(img_pil).enhance(1.5)
    elif transform_name == "saturation_down":
        img_pil = ImageEnhance.Color(img_pil).enhance(0.5)
    elif transform_name == "saturation_up":
        img_pil = ImageEnhance.Color(img_pil).enhance(1.5)
    elif transform_name == "rotate_15":
        img_pil = img_pil.rotate(15)
    elif transform_name == "rotate_-15":
        img_pil = img_pil.rotate(-15)
    elif transform_name == "gaussian_blur":
        img_np = cv2.GaussianBlur(np.array(img_pil), (7, 7), 1.5)
        return img_np
    elif transform_name == "color_shift":
        img_np = np.array(img_pil).astype(np.int16)
        img_np[..., 0] = np.clip(img_np[..., 0] + 20, 0, 255)  # Red
        img_np[..., 1] = np.clip(img_np[..., 1] - 10, 0, 255)  # Green
        return img_np.astype(np.uint8)

    return np.array(img_pil)

def test_model_on_perturbations(model, X_test, y_test):
    """
    Function performs model predictions on test dataset with applied transformations.
    Args:
        model_path (str): Path to saved model.
        X_test (array): Test data.
        y_test (array): Test labels.
    Returns:
        perturbation results (list): List of perturbation results.
    """
    perturbations = [
        "original",
        "brightness_down",
        "brightness_up",
        "contrast_down",
        "contrast_up",
        "saturation_down",
        "saturation_up",
        "rotate_15",
        "rotate_-15",
        "gaussian_blur",
        "color_shift"
    ]

    perturbation_results = []

    for perturb in perturbations:
        if perturb == "original":
            x_mod = X_test
        else:
            x_mod = np.array([apply_transformations(img, perturb) for img in X_test])

        # metrics
        far, frr, precision, recall = evaluate_model(model, x_mod, y_test)

        perturbation_results.append({
            "perturbation": perturb,
            "FAR": round(far, 4),
            "FRR": round(frr, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4)
        })

    return perturbation_results


