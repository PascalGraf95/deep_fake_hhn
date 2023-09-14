from PIL import Image
import os

# Directory containing the images you want to crop
input_directory = "images/scenes/01"


# Function to crop an image to its center square
def crop_to_center_square(image):
    width, height = image.size
    size = min(width, height)
    left = (width - size) // 2
    top = (height - size) // 2
    right = (width + size) // 2
    bottom = (height + size) // 2
    return image.crop((left, top, right, bottom))


if __name__ == '__main__':
    # Loop through all files in the input directory
    for filename in os.listdir(input_directory):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            input_path = os.path.join(input_directory, filename)
            output_path = os.path.join(input_directory, filename)

            # Open the image
            image = Image.open(input_path)

            # Crop the image to its center square
            cropped_image = crop_to_center_square(image)

            # Save the cropped image
            cropped_image.save(output_path)

            print(f'{filename} has been cropped and saved.')

    print('All images have been cropped and saved.')