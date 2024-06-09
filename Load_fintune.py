# import các thư viện
from preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from datasets.simpledatasetloader import SimpleDatasetLoader
from preprocessing.AspectAwarePreprocessor import AspectAwarePreprocesser
from keras.models import load_model
from imutils import paths
import numpy as np
import cv2


# Khởi tạo danh sách nhãn
classLabels = ["daisy", "dandelion", "rose", "sunflower", "tulip"]

print("[INFO] Đang nạp ảnh mẫu để phân lớp (dự đoán)...")
imagePaths = np.array(list(paths.list_images("image"))) #tạo danh sách đường dẫn đến file ảnh trong folder image
idxs = range(0, len(imagePaths)) # Trả về các số nguyên tương ứng với đường dẫn đến file ảnh
imagePaths = imagePaths[idxs] # Tạo danh sách chứa các số nguyên tương ứng với đường dẫn đến file ảnh

sp = AspectAwarePreprocesser(32, 32) # Thiết lập kích thước ảnh 32 x 32
iap = ImageToArrayPreprocessor() # Gọi hàm để chuyển ảnh sang mảng


# Nạp dataset từ đĩa rồi co dãn mức xám của pixel trong vùng [0,1]
sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
data, _ = sdl.load(imagePaths)
data = data.astype("float") / 255.0

# load the pre-trained network
print("[INFO] Nạp model mạng pre-trained ...")
model = load_model("finetune_model.hdf5")

# Dự đoa nhãn cho ảnh đầu vào
print("[INFO] Đang dự đoán để phân lớp...")
preds = model.predict(data, batch_size=32).argmax(axis=1)
# Lặp qua tất cả các file ảnh trong imagePaths
# Nạp ảnh ví dụ --> Vẽ dự  đoán --> Hiển thị ảnh
for (i, imagePath) in enumerate(imagePaths):
    # Đọc file ảnh
    image = cv2.imread(imagePath)
    # Vẽ label dự đoán lên ảnh
    cv2.putText(image, "label: {}".format(classLabels[preds[i]]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    # Hiển thị ảnh
    cv2.imshow("Image", image)
    cv2.waitKey(0)




