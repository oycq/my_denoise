import cv2
import numpy as np
import json

def get_noise_parameters(gain, model_path='noise_model.json'):
    with open(model_path, 'r') as f:
        model = json.load(f)
    # Calculate k (Linear relationship with ISO/gain)
    # k = a * gain + b
    a_k = model['k_coefficients']['a']
    b_k = model['k_coefficients']['b']
    k = a_k * gain + b_k
    # Calculate sigma2 (Quadratic relationship with ISO/gain)
    # sigma^2 = a * gain^2 + b * gain + c
    a_sigma2 = model['sigma2_coefficients']['a']
    b_sigma2 = model['sigma2_coefficients']['b']
    c_sigma2 = model['sigma2_coefficients']['c']
    sigma2 = a_sigma2 * (gain ** 2) + b_sigma2 * gain + c_sigma2

    return k, sigma2

def k_sigma_transform(x, k, sigma2, inverse=False):
    term_bias = sigma2 / (k ** 2)
    if not inverse:
        # Forward transform -> ISO-independent space
        return (x / k) + term_bias
    else:
        # Inverse transform -> Original RAW space
        return k * (x - term_bias)

def add_noise_transformed(transformed_clean):
    # Ensure non-negative values for sqrt calculation
    # In the transformed space, the variance is equal to the pixel value itself.
    noise_variance = np.maximum(transformed_clean, 0)
    noise_std = np.sqrt(noise_variance)
    # Generate random noise
    noise = np.random.normal(0, noise_std, transformed_clean.shape)
    # Add noise
    transformed_noisy = transformed_clean + noise
    return transformed_noisy

# --- Main Execution ---
if __name__ == "__main__":
    filename = '1.raw'
    height = 3000
    width = 4000
    current_gain = 10.0  # Example gain (ISO equivalent)

    # 1. Load Data
    raw_data = np.fromfile(filename, dtype=np.uint16)
    image = raw_data.reshape((height, width))
    image_float = image.astype(np.float32) / 65535.0  # Normalized x*

    # 2. Get Parameters
    k_val, sigma2_val = get_noise_parameters(current_gain)

    # 3. Forward Transform (Clean Image -> ISO-invariant space)
    # We transform the clean image so we can add noise in the normalized space
    transformed_clean = k_sigma_transform(image_float, k_val, sigma2_val, inverse=False)


    # 4. Add Noise (No gain parameter needed here!)
    # The noise added depends entirely on the pixel intensity of transformed_clean
    transformed_noisy = add_noise_transformed(transformed_clean)

    # 5. Inverse Transform (Optional: If you want to view the noisy image in original RAW space)
    # Usually, the network takes the `transformed_noisy` as input directly [cite: 146]
    noisy_raw_image = k_sigma_transform(transformed_noisy, k_val, sigma2_val, inverse=True)

    # Display
    cv2.imshow('Original Clean', image_float)
    cv2.imshow('Synthesized Noisy (RAW Space)', noisy_raw_image) # Normalize for display
    cv2.waitKey(0)
    cv2.destroyAllWindows()