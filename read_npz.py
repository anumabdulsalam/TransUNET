import numpy
import matplotlib.pyplot as plt
data = numpy.load('C:/Users/lab_14/pycharmProjects/MIST-main-v2/data/NIDI/train_npz_new/case0005_slice007.npz')
print(data.files)

image = data['image']
label = data['label']


#white_pixel_value = 1  # Replace this with the actual value representing white
#image = (image == white_pixel_value)
# Display the image
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title('Image')
plt.imshow(image)  # Assuming it's a grayscale image
plt.axis('off')

min_val = numpy.min(image)
max_val = numpy.max(image)
# = (label - min_val) / (max_val - min_val)

# Display the label
plt.subplot(1, 2, 2)
plt.title('Label')
plt.imshow(label, cmap='gray')  # Using a color map for visualization, you might choose a different one based on your data
plt.axis('off')

plt.show()

print(image)



# Get unique values and their counts for the label
unique_label_values, label_counts = numpy.unique(label, return_counts=True)
print("Unique values in label array:", unique_label_values)
print("Counts of unique values in label array:", label_counts)

# Get unique values and their counts for the image
unique_image_values, image_counts = numpy.unique(image, return_counts=True)
print("Unique values in image array:", unique_image_values)
print("Counts of unique values in image array:", image_counts)


print(f"Minimum value in label: {min_val}")
print(f"Maximum value in label: {max_val}")

