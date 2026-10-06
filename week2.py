import os
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as transforms


#trainig on a photo of My baby boy Leo
desktop_path = os.path.expanduser("~/Desktop")
image_path = os.path.join(desktop_path, "Leo.JPG")

# قراءة الصورة باستخدام OpenCV
img = cv2.imread(image_path)

if img is None:
    raise FileNotFoundError(
      
    )

# تحويل نظام الألوان من BGR الخاص بـ OpenCV إلى RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# Preprocessing Pipeline
transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((32, 32)),  # تغيير الحجم لتناسب شبكة CNN
        transforms.ToTensor(),  # تحويل قيم البكسلات إلى Tensor بين [0, 1]
    ]
)

# تطبيق التغييرات وإضافة بُعد الدفعة (Batch Dimension) -> (1, 3, 32, 32)
input_tensor = transform(img_rgb).unsqueeze(0)



# 3 Building (CNN Architecture)

class LeoClassifierCNN(nn.Module):

    def __init__(self, num_classes=2):  # 0 = Cat, 1 = Other
        super(LeoClassifierCNN, self).__init__()

        # طبقات استخراج الخصائص (Feature Extraction)
        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (32x32 -> 16x16)
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (16x16 -> 8x8) #usually it is max pooling 
        )

        # طبقة التصنيف النهائية (Classification Head)
        self.classifier = nn.Linear(32 * 8 * 8, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)  # تسطيح الـ Feature Maps إلى متجه 1D
        logits = self.classifier(x)
        return logits



# 4. تشغيل الصورة داخل النموذج وعرض النتائج

model = LeoClassifierCNN(num_classes=2)
model.eval()  # وضع النموذج في حالة التقييم

with torch.no_grad():
    output_scores = model(input_tensor)
    probabilities = torch.softmax(output_scores, dim=1)

print(f"صورة القط Leo تم قراءتها بنجاح من سطح المكتب!")
print("أبعاد الـ Tensor المدخل للشبكة:", input_tensor.shape)
print("قيم الـ Logits الناتجة:", output_scores.numpy())
print("احتماليات الـ Softmax:", probabilities.numpy())

# عرض صورة Leo
plt.imshow(img_rgb)
plt.title("Leo the Cat")
plt.axis("off")
plt.show()
