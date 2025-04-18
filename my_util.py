import csv
import os

from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

def save_img(tensors):
    for i in range(0,len(tensors[0])):
        pil_image = transforms.ToPILImage()(tensors[i])
        plt.imshow(pil_image)
        plt.savefig('my_image.png')
        plt.savefig(f'/home/lee/PycharmProjects/stageCLIP/demo_img/my_image_{i}.png')


def rename_(path,start):
    file_list = os.listdir(path)
    # a = os.listdir('/home/lee/PycharmProjects/stageCLIP/universal-image-restoration/datasets/groupsc-master_cbsd/datasets/CBSD400')
    file_list = [name for name in file_list if name.endswith(('.bmp', '.png', '.jpg'))]
    file_list = sorted(file_list, key=lambda name: int(''.join(filter(str.isdigit, name))))
    for file_path in file_list:
        # 获取文件所在目录和文件名
        dir_name, old_name = os.path.split(file_path)
        dir_name = path
        # 提取新文件名，保留数字部分及扩展名
        suff = old_name.split('.')[-1]
        # 生成新的文件路径
        new_path = os.path.join(dir_name, str(start) + f'.{suff}')
        old = os.path.join(path,file_path)
        # 使用 os.rename 进行重命名
        os.rename(old, new_path)
        start += 1
        print(f'Renamed: {file_path} -> {new_path}')
def emit_pre2(path):
    file_list = os.listdir(path)
    # a = os.listdir('/home/lee/PycharmProjects/stageCLIP/universal-image-restoration/datasets/groupsc-master_cbsd/datasets/CBSD400')
    file_list = [name for name in file_list if name.endswith(('.bmp', '.png', '.jpg'))]
    for file_path in file_list:
        # 获取文件所在目录和文件名
        dir_name, old_name = os.path.split(file_path)
        # dir_name = path
        # # 提取新文件名，保留数字部分及扩展名
        # suff = old_name.split('.')[-1]
        # # 生成新的文件路径
        new_path = os.path.join(path, old_name[2:])
        # old = os.path.join(path, file_path)
        # 使用 os.rename 进行重命名
        os.rename(os.path.join(path, old_name), new_path)
        # start += 1
        # print(f'Renamed: {file_path} -> {new_path}')
def firstCol2Number(csv_path):
    """
    修改 CSV 文件第一列的内容，将文件路径提取为文件名（去除扩展名），并覆盖保存。

    :param csv_path: CSV 文件路径
    """
    # 创建一个临时列表，用于保存修改后的行数据
    modified_rows = []
    # 打开 CSV 文件进行处理
    with open(csv_path, mode='r', encoding='utf-8') as csv_file:
        reader = csv.reader(csv_file)
        # 提取表头（第一行）
        header = next(reader)
        modified_rows.append(header)  # 保留表头不变

        # 处理其余行
        for row in reader:
            if row and len(row) > 0:  # 确保行不为空
                # 提取第一列内容，获取文件名并去掉后缀
                original_path = row[0]
                base_name = os.path.basename(original_path)  # 提取文件名（带扩展名）
                # file_name = os.path.splitext(base_name)[0]  # 去掉扩展名
                row[0] = base_name  # 修改第一列内容为文件名
            modified_rows.append(row)  # 将修改后的行加入结果

    # 保存修改后的内容到原始 CSV（覆盖保存）
    with open(csv_path, mode='w', encoding='utf-8', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(modified_rows)

    print(f"CSV 文件修改完成，已保存至: {csv_path}")

def firstColconnectDesType(csv_path,des_type):
    """
    修改 CSV 文件第一列的内容，将文件路径提取为文件名（去除扩展名），并覆盖保存。

    :param csv_path: CSV 文件路径
    """
    # 创建一个临时列表，用于保存修改后的行数据
    modified_rows = []
    # 打开 CSV 文件进行处理
    with open(csv_path, mode='r', encoding='utf-8') as csv_file:
        reader = csv.reader(csv_file)
        # 提取表头（第一行）
        header = next(reader)
        modified_rows.append(header)  # 保留表头不变

        # 处理其余行
        for row in reader:
            if row and len(row) > 0:  # 确保行不为空
                # 提取第一列内容，获取文件名并去掉后缀
                original_path = row[0]
                base_name = os.path.basename(original_path)  # 提取文件名（带扩展名）
                # file_name = os.path.splitext(base_name)[0]  # 去掉扩展名
                row[0] = des_type+'/'+base_name  # 修改第一列内容为文件名
            modified_rows.append(row)  # 将修改后的行加入结果

    # 保存修改后的内容到原始 CSV（覆盖保存）
    with open(csv_path, mode='w', encoding='utf-8', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(modified_rows)

    print(f"CSV 文件修改完成，已保存至: {csv_path}")

import cv2
import numpy as np
import os

def synthesize_rainy_image(background_path, rain_streak_path, output_path):
    """
    Synthesizes a rainy image by adding a rain streak layer to a background image.

    Args:
        background_path (str): Path to the clean background image file.
        rain_streak_path (str): Path to the rain streak layer image file (black background).
        output_path (str): Path where the synthesized rainy image will be saved.

    Returns:
        bool: True if synthesis and saving were successful, False otherwise.
    """
    # --- 1. Load Images ---
    print(f"Loading background image: {background_path}")
    background_img = cv2.imread(background_path)

    print(f"Loading rain streak layer: {rain_streak_path}")
    rain_streak_img = cv2.imread(rain_streak_path)

    # --- 2. Validation ---
    if background_img is None:
        print(f"Error: Could not load background image from '{background_path}'")
        print("Please ensure the file exists and is a valid image.")
        return False
    if rain_streak_img is None:
        print(f"Error: Could not load rain streak image from '{rain_streak_path}'")
        print("Please ensure the file exists and is a valid image.")
        return False

    # Check if dimensions match
    if background_img.shape != rain_streak_img.shape:
        print(f"Error: Image dimensions do not match!")
        print(f"Background shape: {background_img.shape}")
        print(f"Rain streak shape: {rain_streak_img.shape}")
        print("Images must have the same height, width, and number of channels.")
        # Optional: Add resizing logic here if needed, but for this dataset
        # they should already match.
        return False

    # Check data types (usually uint8 for PNGs loaded by OpenCV)
    if background_img.dtype != rain_streak_img.dtype:
        print(f"Warning: Image data types differ ({background_img.dtype} vs {rain_streak_img.dtype}).")
        # Attempt to convert rain_streak_img to background_img's type
        try:
            rain_streak_img = rain_streak_img.astype(background_img.dtype)
            print(f"Converted rain streak image to {background_img.dtype}.")
        except Exception as e:
            print(f"Error converting rain streak image type: {e}")
            return False

    print("Image loading and validation successful.")
    print(f"Image shape: {background_img.shape}, Data type: {background_img.dtype}")

    # --- 3. Synthesize using Addition (R = B + S) ---
    # cv2.add performs saturated addition, automatically handling clipping
    # (values > 255 become 255, values < 0 become 0 for uint8)
    print("Performing pixel-wise addition (Background + Rain Streaks)...")
    rainy_img = cv2.add(background_img, rain_streak_img)
    print("Addition complete.")

    # --- Alternative using NumPy (requires manual clipping) ---
    # # Convert to a larger integer type to prevent overflow during addition
    # bg_int = background_img.astype(np.int16)
    # rs_int = rain_streak_img.astype(np.int16)
    # # Add
    # rainy_sum = bg_int + rs_int
    # # Clip the result back to the valid range [0, 255]
    # rainy_clipped = np.clip(rainy_sum, 0, 255)
    # # Convert back to the original data type (uint8)
    # rainy_img_numpy = rainy_clipped.astype(np.uint8)
    # # cv2.add is generally preferred for simplicity and potential optimization

    # --- 4. Save the Result ---
    print(f"Saving synthesized rainy image to: {output_path}")
    try:
        success = cv2.imwrite(output_path, rainy_img)
        if success:
            print("Image saved successfully.")
            return True
        else:
            print(f"Error: Failed to save image to '{output_path}'. Check permissions?")
            return False
    except Exception as e:
        print(f"An error occurred during saving: {e}")
        return False

import cv2
import numpy as np
import os
import argparse # Use argparse for command-line arguments

def synthesize_rainy_image_weight(background_path, rain_streak_path, output_path, temperature_weight=1.0):
    """
    Synthesizes a rainy image by adding a scaled rain streak layer
    to a background image using a temperature weight.

    Args:
        background_path (str): Path to the clean background image file.
        rain_streak_path (str): Path to the rain streak layer image file (black background).
        output_path (str): Path where the synthesized rainy image will be saved.
        temperature_weight (float): Scaling factor for the rain intensity.
                                    1.0 = original intensity, >1.0 = heavier,
                                    <1.0 = lighter, 0.0 = no rain. Defaults to 1.0.

    Returns:
        bool: True if synthesis and saving were successful, False otherwise.
    """
    # --- 1. Load Images ---
    print(f"Loading background image: {background_path}")
    background_img = cv2.imread(background_path)

    print(f"Loading rain streak layer: {rain_streak_path}")
    rain_streak_img = cv2.imread(rain_streak_path)

    # --- 2. Validation ---
    if background_img is None:
        print(f"Error: Could not load background image from '{background_path}'")
        return False
    if rain_streak_img is None:
        print(f"Error: Could not load rain streak image from '{rain_streak_path}'")
        return False

    if background_img.shape != rain_streak_img.shape:
        print(f"Error: Image dimensions do not match!")
        print(f"Background shape: {background_img.shape}")
        print(f"Rain streak shape: {rain_streak_img.shape}")
        return False

    # Ensure images are uint8 for cv2.addWeighted if needed,
    # although imread usually returns uint8 for standard image formats.
    if background_img.dtype != np.uint8:
         print(f"Warning: Background image dtype is {background_img.dtype}, converting to uint8.")
         background_img = background_img.astype(np.uint8) # Or handle appropriately
    if rain_streak_img.dtype != np.uint8:
         print(f"Warning: Rain streak image dtype is {rain_streak_img.dtype}, converting to uint8.")
         rain_streak_img = rain_streak_img.astype(np.uint8)

    print("Image loading and validation successful.")
    print(f"Using temperature weight: {temperature_weight}")

    # --- 3. Synthesize using Weighted Addition ---
    # R = B*1.0 + S*temperature_weight + 0.0
    # cv2.addWeighted handles the scaling and saturation (clipping) correctly.
    print("Performing weighted pixel-wise addition...")
    try:
        rainy_img = cv2.addWeighted(
            src1=background_img,    # First image (background)
            alpha=1.0,              # Weight for the first image
            src2=rain_streak_img,   # Second image (rain streaks)
            beta=temperature_weight,# Weight for the second image (our temperature)
            gamma=0.0               # Scalar added to each sum (offset)
        )
        print("Weighted addition complete.")
    except Exception as e:
        print(f"Error during cv2.addWeighted: {e}")
        print("Ensure temperature_weight is a valid number.")
        return False


    # --- 4. Save the Result ---
    print(f"Saving synthesized rainy image to: {output_path}")
    try:
        # Ensure the output directory exists if specified in the path
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            print(f"Creating output directory: {output_dir}")
            os.makedirs(output_dir)

        success = cv2.imwrite(output_path, rainy_img)
        if success:
            print(f"Image saved successfully: {output_path}")
            return True
        else:
            print(f"Error: Failed to save image to '{output_path}'. Check permissions or path validity.")
            return False
    except Exception as e:
        print(f"An error occurred during saving: {e}")
        return False

import cv2
import numpy as np
import os

def synthesize_snowy_image(background_path, snow_particle_path, output_path, intensity_weight=1.0):
    """
    Synthesizes a snowy image by adding a scaled snow particle layer
    to a background image using an intensity weight.

    This function assumes the snow particle image has snow (white/grey pixels)
    on a black background.

    Args:
        background_path (str): Path to the clean background image file.
        snow_particle_path (str): Path to the snow particle layer image file
                                  (e.g., white/grey particles on a black background).
        output_path (str): Path where the synthesized snowy image will be saved.
        intensity_weight (float): Scaling factor for the snow intensity/visibility.
                                    1.0 = original intensity from the particle image,
                                    >1.0 = more intense/brighter snow,
                                    <1.0 = less intense/fainter snow,
                                    0.0 = no snow. Defaults to 1.0.

    Returns:
        bool: True if synthesis and saving were successful, False otherwise.
    """
    # --- 1. Load Images ---
    print(f"Loading background image: {background_path}")
    background_img = cv2.imread(background_path)

    print(f"Loading snow particle layer: {snow_particle_path}")
    snow_particle_img = cv2.imread(snow_particle_path)

    # --- 2. Validation ---
    if background_img is None:
        print(f"Error: Could not load background image from '{background_path}'")
        return False
    if snow_particle_img is None:
        print(f"Error: Could not load snow particle image from '{snow_particle_path}'")
        return False

    # --- 2a. Handle Dimension Mismatch (Optional but Recommended) ---
    # Check if height and width match. If not, resize the snow layer.
    if background_img.shape[:2] != snow_particle_img.shape[:2]:
        print(f"Warning: Image dimensions do not match! Resizing snow layer.")
        print(f"Background shape: {background_img.shape}")
        print(f"Snow particle shape: {snow_particle_img.shape}")
        # Resize snow layer to match background dimensions (width, height format for cv2.resize)
        target_size = (background_img.shape[1], background_img.shape[0])
        snow_particle_img = cv2.resize(snow_particle_img, target_size, interpolation=cv2.INTER_LINEAR)
        print(f"Resized snow particle shape: {snow_particle_img.shape}")
        # Check if resizing was successful in creating matching shapes (especially channel count might differ)
        if background_img.shape[:2] != snow_particle_img.shape[:2]:
             print(f"Error: Failed to resize snow layer to match background dimensions.")
             return False # Or handle channel differences if necessary


    # --- 2b. Data Type Check ---
    # Ensure images are uint8 for cv2.addWeighted
    if background_img.dtype != np.uint8:
         print(f"Warning: Background image dtype is {background_img.dtype}, converting to uint8.")
         # Be cautious with conversion, ensure value range is appropriate (0-255)
         # background_img = np.clip(background_img, 0, 255).astype(np.uint8) # Example if starting from float
         background_img = background_img.astype(np.uint8) # Assuming it's convertible directly

    if snow_particle_img.dtype != np.uint8:
         print(f"Warning: Snow particle image dtype is {snow_particle_img.dtype}, converting to uint8.")
         snow_particle_img = snow_particle_img.astype(np.uint8)

    print("Image loading and validation successful.")
    print(f"Using snow intensity weight: {intensity_weight}")

    # --- 3. Synthesize using Weighted Addition ---
    # Formula: SnowyImage = Background * alpha + SnowParticles * beta + gamma
    # We want: SnowyImage = Background * 1.0 + SnowParticles * intensity_weight + 0.0
    # cv2.addWeighted handles the scaling and saturation (clipping > 255) correctly.
    print("Performing weighted pixel-wise addition...")
    try:
        # Ensure weight is not negative, which could cause unexpected subtraction
        safe_intensity_weight = max(0.0, intensity_weight)
        if safe_intensity_weight != intensity_weight:
             print(f"Warning: Negative intensity_weight ({intensity_weight}) provided. Clamped to 0.0.")

        snowy_img = cv2.addWeighted(
            src1=background_img,        # First image (background)
            alpha=1.0,                  # Weight for the first image (keep background fully)
            src2=snow_particle_img,     # Second image (snow particles)
            beta=safe_intensity_weight, # Weight for the second image (snow intensity)
            gamma=0.0                   # Scalar added to each sum (offset, usually 0)
        )
        print("Weighted addition complete.")
    except cv2.error as e:
        print(f"Error during cv2.addWeighted: {e}")
        print("Ensure image dimensions and types are compatible after potential resizing/conversion.")
        return False
    except Exception as e:
        print(f"An unexpected error occurred during weighted addition: {e}")
        return False


    # --- 4. Save the Result ---
    print(f"Saving synthesized snowy image to: {output_path}")
    try:
        # Ensure the output directory exists if specified in the path
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            print(f"Creating output directory: {output_dir}")
            os.makedirs(output_dir)

        success = cv2.imwrite(output_path, snowy_img)
        if success:
            print(f"Image saved successfully: {output_path}")
            return True
        else:
            print(f"Error: Failed to save image to '{output_path}'. Check permissions or path validity.")
            return False
    except Exception as e:
        print(f"An error occurred during saving: {e}")
        return False

import cv2
import numpy as np
import os

# Renamed function to clearly distinguish from the addWeighted version
def synthesize_snowy_image_alpha_blend(background_path, snow_particle_path, output_path, intensity_weight=1.0):
    """
    Synthesizes a snowy image using alpha blending to avoid over-exposure.

    Treats the snow particle image (white/grey on black) as both the snow color
    and the basis for its opacity mask. Corrects the over-exposure issue seen
    with simple weighted addition.

    Args:
        background_path (str): Path to the clean background image file.
        snow_particle_path (str): Path to the snow particle layer image file
                                  (e.g., white/grey particles on black background).
        output_path (str): Path where the synthesized snowy image will be saved.
        intensity_weight (float): Controls the opacity/visibility of the snow.
                                    Values around 1.0 are typical. >1 can make
                                    faint snow more opaque, <1 makes snow more
                                    transparent. Clamped between 0 and 1
                                    internally for the alpha value. Defaults to 1.0.

    Returns:
        bool: True if synthesis and saving were successful, False otherwise.
    """
    # --- 1. Load Images ---
    print(f"Loading background image: {background_path}")
    background_img = cv2.imread(background_path)

    print(f"Loading snow particle layer: {snow_particle_path}")
    snow_particle_img_bgr = cv2.imread(snow_particle_path)

    # --- 2. Validation ---
    if background_img is None:
        print(f"Error: Could not load background image from '{background_path}'")
        return False
    if snow_particle_img_bgr is None:
        print(f"Error: Could not load snow particle image from '{snow_particle_path}'")
        return False

    # --- 2a. Handle Dimension Mismatch (Resize Snow Layer) ---
    if background_img.shape[:2] != snow_particle_img_bgr.shape[:2]:
        print(f"Warning: Image dimensions do not match! Resizing snow layer.")
        print(f"Background shape: {background_img.shape}")
        print(f"Snow particle shape: {snow_particle_img_bgr.shape}")
        target_size = (background_img.shape[1], background_img.shape[0]) # (width, height)
        snow_particle_img_bgr = cv2.resize(snow_particle_img_bgr, target_size, interpolation=cv2.INTER_LINEAR)
        print(f"Resized snow particle shape: {snow_particle_img_bgr.shape}")
        if background_img.shape[:2] != snow_particle_img_bgr.shape[:2]:
             print(f"Error: Failed to resize snow layer to match background dimensions.")
             return False

    print("Image loading and validation successful.")
    print(f"Using snow intensity weight (for opacity): {intensity_weight}")

    # --- 3. Prepare for Alpha Blending ---

    # Convert images to float32 for accurate calculations (range 0.0 - 1.0)
    # This prevents clipping during intermediate calculations
    background_fl = background_img.astype(np.float32) / 255.0
    snow_particles_fl = snow_particle_img_bgr.astype(np.float32) / 255.0

    # Create alpha mask from the snow particle image
    # Use grayscale intensity to determine the opacity of each snow particle
    snow_gray = cv2.cvtColor(snow_particle_img_bgr, cv2.COLOR_BGR2GRAY)
    alpha_mask_fl = snow_gray.astype(np.float32) / 255.0 # Normalize to 0.0 - 1.0

    # Apply intensity weight to the alpha mask
    # Clamp values between 0.0 and 1.0 as alpha cannot exceed 1
    effective_alpha = np.clip(alpha_mask_fl * intensity_weight, 0.0, 1.0)

    # Reshape alpha mask to (height, width, 1) for broadcasting across BGR channels
    effective_alpha = effective_alpha[:, :, np.newaxis]

    # --- 4. Perform Alpha Blending ---
    # Formula: output = foreground * alpha + background * (1 - alpha)
    # Where snow is opaque (alpha=1), output = snow_color
    # Where snow is transparent (alpha=0), output = background_color
    print("Performing alpha blending...")
    try:
        # Calculate the background contribution (scaled by 1 - alpha)
        bg_part = background_fl * (1.0 - effective_alpha)

        # Calculate the snow contribution (scaled by alpha)
        snow_part = snow_particles_fl * effective_alpha

        # Combine the parts - this blends the snow onto the background
        blended_fl = bg_part + snow_part

    except Exception as e:
        print(f"Error during alpha blending calculation: {e}")
        return False

    # --- 5. Convert back to uint8 and Save ---
    # Convert float image (0.0-1.0) back to uint8 (0-255)
    snowy_img_final = (blended_fl * 255.0).astype(np.uint8)
    print("Alpha blending complete.")

    print(f"Saving synthesized snowy image to: {output_path}")
    try:
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            print(f"Creating output directory: {output_dir}")
            os.makedirs(output_dir)

        success = cv2.imwrite(output_path, snowy_img_final)
        if success:
            print(f"Image saved successfully: {output_path}")
            return True
        else:
            print(f"Error: Failed to save image to '{output_path}'. Check permissions or path validity.")
            return False
    except Exception as e:
        print(f"An error occurred during saving: {e}")
        return False

# --- Example Usage ---
# Replace with your actual file paths
# background_image_file = 'image1_city.jpg'
# snow_layer_file = 'image2_snow_particles.png' # White/grey on black
# output_snowy_file = 'output/city_snowy_alpha_blended.jpg'

# Create output dir if needed
# if not os.path.exists('output'):
#      os.makedirs('output')

# if os.path.exists(background_image_file) and os.path.exists(snow_layer_file):
#     # Use the alpha blending function
#     # Adjust intensity_weight (opacity factor) - start around 1.0
#     success = synthesize_snowy_image_alpha_blend(
#         background_image_file,
#         snow_layer_file,
#         output_snowy_file,
#         intensity_weight=1.0 # <-- Adjust this value (e.g., 0.8, 1.0, 1.2)
#     )
#     if success:
#         print("Snowy image created successfully using alpha blending.")
#     else:
#         print("Failed to create snowy image.")
# else:
#     print("Error: Input image files not found.")

import os

import cv2
import numpy as np
import matplotlib.pyplot as plt

def add_haze(image_path, output_path, beta=3.5, airlight_rgb=(210, 215, 220), simulate_depth=True, depth_map_path=None):
    # --- 1. Read Input Image ---
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image from {image_path}")
        return None

    # Normalize image to [0, 1] float for calculations
    img_float = img.astype(np.float32) / 255.0
    height, width, _ = img.shape

    # --- 2. Get Depth Map d(x) ---
    if simulate_depth:
        print("Simulating depth map (linear gradient top-to-bottom)...")
        # Create a simple linear depth map: depth increases from bottom (0) to top (1)
        # More sophisticated simulations could be used (e.g., radial, based on simple features)
        y_coords = np.arange(height)
        # Normalize depth to roughly [0, 1] range (can be scaled later if needed)
        depth_values = (1.0 - y_coords / (height - 1)) if height > 1 else np.zeros(height)
        # Tile this across the width to create a 2D depth map
        depth_map = np.tile(depth_values, (width, 1)).T # Shape (height, width)
        print(f"Simulated depth map range: {np.min(depth_map)} to {np.max(depth_map)}")

    elif depth_map_path:
        print(f"Reading depth map from {depth_map_path}...")
        depth_map_gray = cv2.imread(depth_map_path, cv2.IMREAD_GRAYSCALE)
        if depth_map_gray is None:
            print(f"Error: Could not read depth map from {depth_map_path}. Falling back to simulation.")
            # Fallback to simulation
            y_coords = np.arange(height)
            depth_values = (1.0 - y_coords / (height - 1)) if height > 1 else np.zeros(height)
            depth_map = np.tile(depth_values, (width, 1)).T
        else:
            # Resize depth map to match image dimensions if necessary
            if depth_map_gray.shape != (height, width):
                depth_map_gray = cv2.resize(depth_map_gray, (width, height), interpolation=cv2.INTER_LINEAR)
            # Normalize depth map to [0, 1] (assuming white=far=max_depth, black=near=0)
            depth_map = depth_map_gray.astype(np.float32) / 255.0
            print(f"Loaded depth map range: {np.min(depth_map)} to {np.max(depth_map)}")
    else:
         print("Error: Need either simulate_depth=True or a valid depth_map_path.")
         return None

    # --- 3. Calculate Transmission Map t(x) ---
    # t(x) = exp(-beta * d(x))
    # Ensure beta is positive
    beta = abs(beta)
    transmission = np.exp(-beta * depth_map)

    # Add channel dimension for broadcasting: (height, width) -> (height, width, 1)
    transmission_rgb = np.expand_dims(transmission, axis=2)

    # --- 4. Define Atmospheric Light A ---
    airlight = np.array(airlight_rgb).astype(np.float32) / 255.0
    # Reshape for broadcasting: (3,) -> (1, 1, 3)
    airlight = airlight.reshape(1, 1, 3)

    # --- 5. Apply Atmospheric Scattering Model ---
    # I(x) = J(x) * t(x) + A * (1 - t(x))
    hazy_img_float = img_float * transmission_rgb + airlight * (1 - transmission_rgb)

    # --- 6. Post-processing and Save ---
    # Clip values to ensure they are in [0, 1] range
    hazy_img_float = np.clip(hazy_img_float, 0, 1)

    # Convert back to uint8 [0, 255]
    hazy_img_uint8 = (hazy_img_float * 255).astype(np.uint8)

    # Save the result
    cv2.imwrite(output_path, hazy_img_uint8)
    print(f"Hazy image saved to {output_path}")

    # --- Optional: Display results ---
    # plt.figure(figsize=(18, 6))
    #
    # plt.subplot(1, 3, 1)
    # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # plt.title('Original Image (J)')
    # plt.axis('off')
    #
    # plt.subplot(1, 3, 2)
    # # Display transmission map (easier to visualize than raw depth sometimes)
    # transmission_display = plt.imshow(transmission, cmap='gray')
    # plt.title(f'Transmission Map (t), beta={beta:.2f}')
    # plt.colorbar(transmission_display)
    # plt.axis('off')
    #
    # plt.subplot(1, 3, 3)
    # plt.imshow(cv2.cvtColor(hazy_img_uint8, cv2.COLOR_BGR2RGB))
    # plt.title(f'Synthesized Hazy Image (I)')
    # plt.axis('off')
    #
    # plt.tight_layout()
    # plt.show()
    #
    # return hazy_img_uint8


# --- Main Execution with Command-Line Arguments ---
if __name__ == "__main__":
    #TODO add rain
    # parser = argparse.ArgumentParser(description="Synthesize a rainy image by adding weighted rain streaks.")
    # parser.add_argument("-bg", "--background", type=str, default="image4.png",
    #                     help="Path to the clean background image (default: image4.png)")
    # parser.add_argument("-rs", "--rainstreak", type=str, default="image1.png",
    #                     help="Path to the rain streak layer image (default: image1.png)")
    # parser.add_argument("-o", "--output", type=str, default="rainy_synthesized_weighted_output.png",
    #                     help="Path for the output synthesized rainy image (default: rainy_synthesized_weighted_output.png)")
    # parser.add_argument("-w", "--weight", type=float, default=1,
    #                     help="Temperature weight for rain intensity (0.0=none, 1.0=original, >1.0=heavier, default: 1.0)")
    #
    # args = parser.parse_args()
    #
    # # Check if input files exist
    # if not os.path.exists(args.background):
    #     print(f"Fatal Error: Input background file '{args.background}' not found.")
    # elif not os.path.exists(args.rainstreak):
    #     print(f"Fatal Error: Input rain streak file '{args.rainstreak}' not found.")
    # else:
    #     # Run the synthesis process
    #     synthesize_rainy_image_weight(args.background, args.rainstreak, args.output, args.weight)
    #************************************
    #TODO trim
    # firstCol2Number('/home/lee/PycharmProjects/stageCLIP/universal-image-restoration/datasets/universal/train/image_descriptions_5144.csv')
    # firstCol2Number('/home/lee/PycharmProjects/stageCLIP/universal-image-restoration/datasets/universal/train/image_descriptions_rainL.csv')
    # firstCol2Number('/home/lee/PycharmProjects/stageCLIP/universal-image-restoration/datasets/universal/train/image_descriptions_noisy_cbsd400.csv')
    #************************************
    #TODO append des type
    # firstColconnectDesType('/home/lee/PycharmProjects/stageCLIP/universal-image-restoration/datasets/universal/train/image_descriptions_snow_400_universal.csv','snow')
    #########################
    # emit_pre2('/home/lee/PycharmProjects/stageCLIP/universal-image-restoration/datasets/universal/val/rain/GT')
    # rename_('/home/lee/PycharmProjects/stageCLIP/universal-image-restoration/datasets/groupsc-master_cbsd/datasets/CBSD400',4745)
    #TODO hazy
    # --- How to Use ---
    # 1. Save your input image (图1) as 'image1.jpg' (or update the path)
    # input_image = '/home/lee/PycharmProjects/stageCLIP/universal-image-restoration/datasets/universal/train/hazy/GT/200.jpg'
    # output_image = '/home/lee/PycharmProjects/stageCLIP/universal-image-restoration/datasets/universal/train/hazy/GT/194_hazy.jpg'
    #
    # # 2. *** Tune these parameters *** to match 图2 visually:
    # # Beta (β): Controls haze density. Higher means more haze. Start around 1.0-2.0 and adjust.
    # # Based on 图2, the haze is quite significant, especially in the distance.
    # #TODO 1.0，1.5，2.0
    # beta_value = 2.0  # <-- TRY ADJUSTING THIS (e.g., 1.0, 1.5, 2.0, 2.5)
    # hazy_image = add_haze(input_image, output_image, beta=beta_value)
    #
    # # If you had a real depth map (e.g., 'depth.png'):
    # # hazy_image = add_haze(input_image, output_image, beta=beta_value, airlight_rgb=airlight_color, simulate_depth=False, depth_map_path='depth.png')
    #
    # if hazy_image is not None:
    #     print("Haze simulation complete.")
    #TODO snow
    # synthesize_snowy_image(
    #     '/home/lee/PycharmProjects/stageCLIP/universal-image-restoration/datasets/SRRS/SRRS-2021/gt/46.jpg',
    #     # '/home/lee/PycharmProjects/stageCLIP/universal-image-restoration/datasets/SRRS/SRRS-2021/test/snow_4.jpg',
    #     # '/home/lee/PycharmProjects/stageCLIP/universal-image-restoration/datasets/SRRS/SRRS-2021/Small Snow/46.jpg',
    #     # '/home/lee/PycharmProjects/stageCLIP/universal-image-restoration/datasets/SRRS/SRRS-2021/Mid Snow/46.jpg',
    #     # '/home/lee/PycharmProjects/stageCLIP/universal-image-restoration/datasets/SRRS/SRRS-2021/Big Snow/46.jpg',
    #     '/home/lee/PycharmProjects/stageCLIP/universal-image-restoration/datasets/SRRS/SRRS-2021/Combine Snow/46.jpg',
    #     '/home/lee/PycharmProjects/stageCLIP/universal-image-restoration/datasets/SRRS/SRRS-2021/test/snow_4.jpg',
    # 0.5)

    pass



