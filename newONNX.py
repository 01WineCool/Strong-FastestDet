import cv2
import numpy as np
import onnxruntime as ort
import time
from collections import deque


class FastestDetRaspberry:
    def __init__(self, onnx_path, class_names, input_size=320):
        # 初始化ONNX运行时
        self.session = ort.InferenceSession(onnx_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_inputs()[0].name
        self.class_names = class_names
        self.input_size = input_size

        # FPS计算
        self.fps_buffer = deque(maxlen=10)
        self.prev_time = time.time()

        # 初始化CSI摄像头
        self.cap = self.init_csi_camera()

    def init_csi_camera(self):
        """初始化树莓派CSI摄像头"""
        # CSI摄像头配置参数
        gstreamer_pipeline = (
            "nvarguscamerasrc ! "
            "video/x-raw(memory:NVMM), width=640, height=480, format=NV12, framerate=30/1 ! "
            "nvvidconv flip-method=0 ! "
            "video/x-raw, width=640, height=480, format=BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=BGR ! appsink"
        )
        return cv2.VideoCapture(gstreamer_pipeline, cv2.CAP_GSTREAMER)

    def preprocess(self, image):
        """图像预处理"""
        # 调整大小并归一化
        img = cv2.resize(image, (self.input_size, self.input_size))
        img = img.astype(np.float32) / 255.0
        # HWC -> CHW
        img = np.transpose(img, (2, 0, 1))
        # 添加batch维度
        img = np.expand_dims(img, axis=0)
        return img

    def postprocess(self, outputs, original_img):
        """后处理输出"""
        boxes, scores, classes = [], [], []
        original_h, original_w = original_img.shape[:2]

        # 解析输出 (根据你的模型输出格式调整)
        # 这里假设输出是[1, num_preds, 85]格式
        outputs = np.squeeze(outputs)

        for pred in outputs:
            # 前4个是box坐标 (cx, cy, w, h)
            cx, cy, w, h = pred[:4]
            # 然后是置信度和类别概率
            conf = pred[4]
            class_probs = pred[5:]

            if conf < 0.5:  # 置信度阈值
                continue

            # 获取类别
            class_id = np.argmax(class_probs)

            # 转换box坐标到原始图像尺寸
            x1 = int((cx - w / 2) * original_w)
            y1 = int((cy - h / 2) * original_h)
            x2 = int((cx + w / 2) * original_w)
            y2 = int((cy + h / 2) * original_h)

            boxes.append([x1, y1, x2, y2])
            scores.append(conf)
            classes.append(class_id)

        # NMS处理
        indices = cv2.dnn.NMSBoxes(boxes, scores, 0.5, 0.4)

        results = []
        for i in indices:
            i = i[0] if isinstance(i, (list, np.ndarray)) else i
            results.append({
                'box': boxes[i],
                'score': scores[i],
                'class_id': classes[i],
                'class_name': self.class_names[classes[i]]
            })

        return results

    def draw_detections(self, image, detections):
        """在图像上绘制检测结果"""
        for det in detections:
            x1, y1, x2, y2 = det['box']
            label = f"{det['class_name']} {det['score']:.2f}"

            # 绘制矩形框
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 绘制标签背景
            (label_width, label_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(
                image,
                (x1, y1 - label_height - 10),
                (x1 + label_width, y1),
                (0, 255, 0),
                cv2.FILLED
            )

            # 绘制文本
            cv2.putText(
                image, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
            )

        # 显示FPS
        current_time = time.time()
        fps = 1 / (current_time - self.prev_time)
        self.prev_time = current_time
        self.fps_buffer.append(fps)
        avg_fps = sum(self.fps_buffer) / len(self.fps_buffer)

        cv2.putText(
            image, f"FPS: {avg_fps:.1f}", (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
        )

        return image

    def run(self):
        """主循环"""
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break

                # 预处理
                input_tensor = self.preprocess(frame)

                # 推理
                outputs = self.session.run(
                    [self.output_name],
                    {self.input_name: input_tensor}
                )[0]

                # 后处理
                detections = self.postprocess(outputs, frame)

                # 绘制结果
                result_frame = self.draw_detections(frame.copy(), detections)

                # 显示结果
                cv2.imshow("FastestDet - Raspberry Pi", result_frame)

                # 退出键
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            self.cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    # 替换为你的类别名称
    CLASS_NAMES = ["person", "car", "dog", "cat"]  # 根据你的数据集修改

    # 使用量化后的模型
    detector = FastestDetRaspberry(
        onnx_path="FastestDet_raspberry_quant.onnx",
        class_names=CLASS_NAMES,
        input_size=320  # 与导出时一致
    )

    detector.run()