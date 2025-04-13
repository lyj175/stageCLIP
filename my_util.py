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
    #************************************
    emit_pre2('/home/lee/PycharmProjects/stageCLIP/universal-image-restoration/datasets/universal/val/rain/GT')
    # rename_('/home/lee/PycharmProjects/stageCLIP/universal-image-restoration/datasets/groupsc-master_cbsd/datasets/CBSD400',4745)
