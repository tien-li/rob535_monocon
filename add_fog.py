import cv2
import numpy as np
import os
import shutil

def adjust_brightness_contrast(image_path, brightness=0, contrast=0):
    img = cv2.imread(image_path)
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow
        
        buf = cv2.addWeighted(img, alpha_b, img, 0, gamma_b)
    else:
        buf = img.copy()

    if contrast != 0:
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)
        
        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf

# Function to apply fog effect
def apply_fog_effect(img, fog_density=0.005):
    height, width = img.shape[:2]
    whiteness = 30  # Fog is usually white
    overlay = np.full((height, width, 3), whiteness, dtype='uint8')  # Create white overlay
    visibility = 1 - np.exp(-fog_density * np.arange(height))  # Calculate visibility for each row
    visibility = (1 - visibility[:, np.newaxis])  # Invert visibility

    foggy_img = np.zeros_like(img)
    for i in range(height):
        alpha = float(visibility[height - 1 - i])
        foggy_img[i, :] = cv2.addWeighted(img[i, :], alpha, overlay[i, :], 1 - alpha, 0)
    return foggy_img

def add_fog_with_gradient(image_path, intensity=0.9):
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read the image {image_path}.")
        return None

    # Create a gradient fog layer
    rows, cols, channels = img.shape
    fog_layer = np.zeros_like(img, dtype=img.dtype)

    # Generate a vertical gradient
    for i in range(rows):
        alpha = intensity * (1 - (i / rows))  # More intense at the top
        fog_layer[i, :, :] = np.full((cols, channels), 255 * alpha, dtype=img.dtype)

    # Blend the fog layer with the original image
    return cv2.addWeighted(img, 1 - intensity, fog_layer, intensity, 0)

def add_fog(image_path, intensity=0.5):
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read the image {image_path}.")
        return None

    # Create a white fog layer
    fog_layer = np.full(img.shape, 128, dtype=img.dtype)

    # Blend the fog layer with the original image
    return cv2.addWeighted(img, 1 - intensity, fog_layer, intensity, 0)

def process_images(src_image_dir, src_label_dir, src_calib_dir, dst_image_dir, dst_label_dir, dst_calib_dir, intensities):
    # Create destination directories if they don't exist
    os.makedirs(dst_image_dir, exist_ok=True)
    os.makedirs(dst_label_dir, exist_ok=True)
    os.makedirs(dst_calib_dir, exist_ok=True)

    train_images = []
    calib_images = []

    # Iterate over all images in the source directory
    for filename in os.listdir(src_image_dir):
        if filename.endswith('.png'):
            image_path = os.path.join(src_image_dir, filename)

            # Create foggy images for each intensity level
            for intensity in intensities:
                # foggy_img = add_fog(image_path, intensity)
                # foggy_img = add_fog_with_gradient(image_path, intensity)

                foggy_img = adjust_brightness_contrast(image_path, brightness=-100, contrast=-50)
                foggy_img = apply_fog_effect(foggy_img, fog_density=0.002)


                if foggy_img is not None:
                    # foggy_image_name = f"{int(intensity * 10)}" + filename.split('.')[0][1:]
                    foggy_image_name = filename.split('.')[0]
                    foggy_image_name = f"{foggy_image_name}.png"
                    cv2.imwrite(os.path.join(dst_image_dir, foggy_image_name), foggy_img)

                    # Add to train and calib lists
                    image_name_without_extension = foggy_image_name.split('.')[0]
                    train_images.append(image_name_without_extension)
                    calib_images.append(image_name_without_extension)


                    # Copy the corresponding label file
                    label_file = os.path.join(src_label_dir, filename.replace('.png', '.txt'))
                    if os.path.exists(label_file):
                        # print(label_file)
                        shutil.copy(label_file, os.path.join(dst_label_dir, foggy_image_name.replace('.png', '.txt')))

                    # Copy the corresponding calib file
                    calib_file = os.path.join(src_calib_dir, filename.replace('.png', '.txt'))
                    if os.path.exists(calib_file):
                        # print(calib_file)
                        shutil.copy(calib_file, os.path.join(dst_calib_dir, foggy_image_name.replace('.png', '.txt')))

    # Write train and calib lists to files
    with open('KITTI_fog/training/ImageSets/train.txt', 'w') as f:
        for item in train_images:
            if int(item[-4:]) <= 927:
                f.write("%s\n" % item)

    with open('KITTI_fog/training/ImageSets/val.txt', 'w') as f:
        for item in train_images:
            if int(item[-4:]) > 927:
                f.write("%s\n" % item)


    with open('KITTI_fog/training/ImageSets/calib.txt', 'w') as f:
        for item in calib_images:
            f.write("%s\n" % item)


def main():
    os.makedirs('KITTI_fog/training', exist_ok=True)
    os.makedirs('KITTI_fog/training/ImageSets', exist_ok=True)

    src_image_dir = 'KITTI/training/image_2'
    src_label_dir = 'KITTI/training/label_2'
    src_calib_dir = 'KITTI/training/calib'
    dst_image_dir = 'KITTI_fog/training/image_2'
    dst_label_dir = 'KITTI_fog/training/label_2'
    dst_calib_dir = 'KITTI_fog/training/calib'
    intensities = [0, 0.7]

    process_images(src_image_dir, src_label_dir, src_calib_dir, dst_image_dir, dst_label_dir, dst_calib_dir, intensities)

if __name__ == '__main__':
    main()