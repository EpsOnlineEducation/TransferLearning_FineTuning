# import các thư viện
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from preprocessing.AspectAwarePreprocessor import AspectAwarePreprocesser
from datasets.simpledatasetloader import SimpleDatasetLoader
from keras.layers import Input
from keras.applications.vgg16 import VGG16
from tensorflow.keras.optimizers import SGD
from keras.models import Model
from imutils import paths
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.layers.core import Dense
import numpy as np
import os


# Nạp ảnh từ đĩa
print("[INFO] Đang nạp ảnh...")
imagePaths = list(paths.list_images("datasets")) # tạo danh sách đường dẫn đến các folder con của folder datasets
classNames = [pt.split(os.path.sep)[-2] for pt in imagePaths]
classNames = [str(x) for x in np.unique(classNames)]

# Tiền xử lý ảnh
aap = AspectAwarePreprocesser(32, 32) # Điều chỉnh kích thước sau khi tăng cường dữ liệu ảnh
iap= ImageToArrayPreprocessor()
sdl = SimpleDatasetLoader(preprocessors=[aap, iap])
(data,labels)  = sdl.load(imagePaths,verbose=500)
data = data.astype("float") / 255.0

# Phân chia dữ liệu
(Data_train,Label_train,Data_test,Label_test) = train_test_split(data,labels,test_size=0.25,random_state=42)
# Mã hóa Label
Label_train = LabelBinarizer().fit_transform(Label_train)
Label_test = LabelBinarizer().fit_transform(Label_test)

# Tải mạng VGG16 và loại bỏ lớp kết nối đầy đủ của mô hình (tham số include_top=False)
baseLayer = VGG16(weights='imagenet',include_top=False,input_tensor=Input(shape = (32,32,3)))
# Định nghĩa lớp Top (head) mới, có 2 lớp FC (VGG có 3 lớp FC)
NewTop = baseLayer.output
NewTop = Flatten(name='flatten')(NewTop)
NewTop = Dense(256, activation='relu')(NewTop)
NewTop = Dropout(0.5)(NewTop)
NewTop = Dense(5, activation='softmax')(NewTop) # Thêm một lớp softmax cho 5 nhãn
# Nối lớp Top mới vào mô hình VGG16
model = Model(inputs=baseLayer.input,outputs = NewTop)
# Duyệt qua tất cả các lớp và đóng băng các lớp cơ sở
# để không làm thay đổi các đặc trưng đã được học trước đó
for layer in baseLayer.layers:
    layer.trainable = False
# Biên dịch mô hình
print("[INFO] Đang biên dịch mô hình....")
opt = SGD(learning_rate = 0.001)
model.compile(loss = 'categorical_crossentropy',optimizer = opt,metrics =['accuracy'])
# Train mô hình. Chú ý: chỉ train lớp mới thêm vào
print("[INFO] training head...")
model.fit(Data_train,Label_train,batch_size = 32,validation_data = (Data_test,Label_test),epochs=3, verbose = 1)

# Đánh giá mô hình
print("[INFO] Đang đánh giá ...")
predictions = model.predict(Data_test,batch_size=32)
print(classification_report(Label_test.argmax(axis =1), predictions.argmax(axis =1),target_names=classNames))

# Sau khi lắp vào các lớp Top, tinh chỉnh mô hình (có thể mở một số hoặc tất cả các lớp cơ sở)
for layer in baseLayer.layers[15:]:    # Mở từ lớp cơ sở thứ 15
    layer.trainable = True
# Biên dịch lại mô hình
print("[INFO] Đang biên dịch lại mô hình ...")
opt = SGD(learning_rate=0.001)
model.compile(loss = 'categorical_crossentropy',optimizer = opt,metrics=['accuracy'])
print("[INFO] Đang fine-tuning mô hình...")
model.fit(Data_train,Label_train,batch_size=32, validation_data = (Data_test,Label_test),epochs = 5,verbose = 1)
# Đánh giá kết quả của mô hình sau khi tinh chỉnh
print("[INFO] Đánh giá mô hình sau khi tinh chỉnh...")
predictions = model.predict(Data_test,batch_size=32)
print(classification_report(Label_test.argmax(axis =1), predictions.argmax(axis =1),target_names=classNames))
# Lưu mô hình vào đĩa
print("[INFO] Lưu mô hình...")
model.save("finetune_model.hdf5")

# Nếu muốn vẽ biểu đồ thì thêm lệnh ở đây


