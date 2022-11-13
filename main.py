import cv2
from PIL import Image

def edge_mask(image):

    # Grayscale
    edge_mask = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # smooth
    for i in range(0, 50):
        edge_mask = cv2.medianBlur(edge_mask, 9)

    # edge detection
    edge_mask = cv2.adaptiveThreshold(edge_mask, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2.7)

    return edge_mask

def color_reduce(filename, reducer):
    
    image = Image.open(filename)

    print("[*] réduction de la palette de couleurs..")
    image = image.quantize(reducer)

    image.save("reduce.png")

    img_reduce = cv2.imread("reduce.png")

    return img_reduce


def min_filter(image, k_size):

    # kernel
    size = (k_size,k_size)
    shape = cv2.MORPH_RECT
    kernel = cv2.getStructuringElement(shape, size)

    # on prend le maximum
    rslt = cv2.erode(image, kernel)

    return rslt



# on charge l'image
original_image = cv2.imread("test.jpg")

# on floute l'originale pour réduire la variété de pixels
print("[*] génération du masque de bords..")

edge_mask_img = cv2.medianBlur(original_image, 9)
edge_mask_img = edge_mask(edge_mask_img)

print("[*] triage des nombres maximum..")
edge_mask_img = min_filter(edge_mask_img, 24)

cv2.imwrite("edge.jpg", edge_mask_img)


# traitement intermédiaire
img_reduce = cv2.blur(original_image, (150,150), 5)
img_reduce = min_filter(img_reduce, 64)

cv2.imwrite("blur.jpg", img_reduce)

img_reduce = color_reduce("blur.jpg", 128)



# on bitwise
cartoon = cv2.bitwise_and(img_reduce, img_reduce, mask=edge_mask_img)


cv2.imwrite("cartoon.png", cartoon)