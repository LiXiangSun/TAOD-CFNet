def ODGA(img):
    img = np.array(img)
    maxV = np.max(img)
    ET = maxV / 2
    Guss = 0
    sum = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] > ET:
                sum = sum + img[i, j]
                Guss = Guss + 1
                img[i, j] = img[i, j]
    AvgSum = sum / Guss
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] > AvgSum:
                img[i, j] = img[i, j]

    img = Image.fromarray(np.uint8(img))
    return img