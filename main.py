import cv2
import torch
from ultralytics import YOLO

# 检查并使用GPU或CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# 加载模型并将其移动到设备
model = YOLO('./models/best.pt')
model.to(device)

# 开启摄像头捕捉
# cap = cv2.VideoCapture("Learn American Sign Language Letters.mp4")

cap = cv2.VideoCapture(0)

# 设置置信度阈值
confidence_threshold = 0.25

keep_running = True

while keep_running:
    ret, frame = cap.read()
    if not ret:
        break

    # 使用模型进行预测
    results = model(frame, conf = confidence_threshold)

    # 找到最高置信度的检测结果
    best_result = None
    max_conf = 0

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # 获取边界框
        confs = result.boxes.conf.cpu().numpy()  # 获取置信度
        labels = result.boxes.cls.cpu().numpy()  # 获取分类标签
        
        for box, conf, label in zip(boxes, confs, labels):
            if conf > max_conf:
                max_conf = conf
                best_result = (box, conf, label)

    # 绘制最高置信度的检测结果
    if best_result is not None:
        box, conf, label = best_result
        x1, y1, x2, y2 = map(int, box)
        label_text = f'{model.names[int(label)]} {conf:.2f}'
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # 显示帧
    cv2.imshow('YOLOv8 Detection', frame)

    # 检查是否按下 't' 键以终止
    if cv2.waitKey(1) & 0xFF == ord('t'):
        keep_running = False
        break

# 释放摄像头和关闭窗口
cap.release()
cv2.destroyAllWindows()