import argparse
import time
import cv2
import numpy as np
import onnxruntime as ort
from threading import Thread


class CSI_Camera_Detector:
    def __init__(self, args):
        # 硬件初始化
        self.init_camera(args.source)

        # 模型加载
        self.init_model(args.model, args.classes)

        # 性能监控
        self.frame_count = 0
        self.fps = 0
        self.prev_time = 0

        # 显示参数
        self.view_img = args.view_img
        self.line_thickness = 2  # 检测框线宽

    def init_camera(self, source):
        """初始化CSI摄像头或TCP视频流"""
        if source.startswith('tcp'):
            # TCP视频流模式（通过rpicam-vid转发）
            self.cap = cv2.VideoCapture(source)
        else:
            # 本地CSI摄像头直连模式
            self.cap = cv2.VideoCapture(self.gstreamer_pipeline(), cv2.CAP_GSTREAMER)

        assert self.cap.isOpened(), "摄像头初始化失败，请检查：\n1. 摄像头连接\n2. 用户权限\n3. 是否启用摄像头(sudo raspi-config)"

    def gstreamer_pipeline(self):
        """CSI摄像头专用GStreamer管道配置"""
        return (
            'nvarguscamerasrc ! '
            'video/x-raw(memory:NVMM), '
            'width=640, height=480, '
            'format=NV12, framerate=30/1 ! '
            'nvvidconv flip-method=0 ! '
            'videoconvert ! '
            'appsink max-buffers=1 drop=True'
        )

    def init_model(self, model_path, classes_path):
        """加载ONNX模型和类别信息"""
        # ONNX Runtime配置（启用多线程）
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4  # 并行计算线程数

        self.session = ort.InferenceSession(model_path, options)
        self.input_name = self.session.get_inputs()[0].name

        # 加载类别标签
        with open(classes_path) as f:
            self.classes = [line.strip() for line in f.readlines()]

        # 获取模型输入尺寸（修正类型转换）
        model_metadata = self.session.get_inputs()[0]
        self.input_shape = [int(dim) for dim in model_metadata.shape[2:]]  # 转换为int列表
        self.channels = int(model_metadata.shape[1])  # 通道数也需转换

        print(f"✓ 模型加载成功 | 输入尺寸: {self.input_shape} | 类别数: {len(self.classes)}")

    def preprocess(self, frame):
        """图像预处理流水线"""
        # 保持长宽比的缩放
        h, w = frame.shape[:2]
        scale = min(self.input_shape[0] / h, self.input_shape[1] / w)
        resized = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)

        # 填充至模型输入尺寸
        padded = np.full((self.input_shape[0], self.input_shape[1], 3), 114, dtype=np.uint8)
        padded[:resized.shape[0], :resized.shape[1]] = resized

        # 转换为模型输入格式
        input_data = padded.transpose(2, 0, 1)  # HWC -> CHW
        input_data = np.ascontiguousarray(input_data, dtype=np.float32) / 255.0
        return np.expand_dims(input_data, axis=0)  # 添加batch维度

    def postprocess(self, outputs, origin_shape):
        """后处理（包含NMS和坐标转换）"""
        predictions = np.squeeze(outputs[0])  # 假设输出形状为(num_detections, 6+num_classes)
        conf_threshold = 0.5
        iou_threshold = 0.5
        detections = []

        # 解析每个检测结果
        for det in predictions:
            if len(det) < 5 + len(self.classes):
                continue

            # 提取坐标信息 (xywh格式)
            x_center, y_center, w, h = det[:4]
            obj_conf = float(det[4])

            # 获取类别概率
            class_probs = det[5:5 + len(self.classes)]
            class_id = np.argmax(class_probs)
            class_prob = class_probs[class_id]
            conf = obj_conf * class_prob

            if conf < conf_threshold:
                continue

            # 转换为xyxy格式
            x1 = float(x_center - w / 2)
            y1 = float(y_center - h / 2)
            x2 = float(x_center + w / 2)
            y2 = float(y_center + h / 2)

            detections.append([x1, y1, x2, y2, conf, class_id])

        # 非极大值抑制 (NMS)
        if len(detections) > 0:
            detections = np.array(detections)
            boxes = detections[:, :4]
            scores = detections[:, 4]

            # 使用OpenCV的NMS
            indices = cv2.dnn.NMSBoxes(
                bboxes=boxes.tolist(),
                scores=scores.tolist(),
                score_threshold=conf_threshold,
                nms_threshold=iou_threshold
            )
            if len(indices) > 0:
                detections = detections[indices.flatten()]
            else:
                detections = np.zeros((0, 6))
        else:
            detections = np.zeros((0, 6))

        # 坐标缩放至原始图像尺寸
        scale = min(self.input_shape[0] / origin_shape[0], self.input_shape[1] / origin_shape[1])
        if len(detections) > 0:
            detections[:, :4] /= scale

        return detections

    def draw_boxes(self, frame, detections):
        """绘制检测框和性能指标"""
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            label = f'{self.classes[int(cls)]} {conf:.2f}'

            # 绘制边界框
            cv2.rectangle(frame,
                          (int(x1), int(y1)),
                          (int(x2), int(y2)),
                          (0, 255, 0), self.line_thickness)

            # 显示标签背景
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame,
                          (int(x1), int(y1) - text_height - 10),
                          (int(x1) + text_width, int(y1)),
                          (0, 0, 255), -1)

            # 显示标签文本
            cv2.putText(frame, label,
                        (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 显示FPS
        cv2.putText(frame, f'FPS: {self.fps:.1f}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return frame

    def run(self):
        """主检测循环"""
        try:
            while True:
                # 读取帧
                ret, frame = self.cap.read()
                if not ret:
                    print("视频流中断，正在重连...")
                    time.sleep(1)
                    self.init_camera(args.source)
                    continue

                # 预处理
                input_tensor = self.preprocess(frame)

                # 推理
                outputs = self.session.run(None, {self.input_name: input_tensor})

                # 后处理
                detections = self.postprocess(outputs, frame.shape[:2])

                # 显示
                if self.view_img:
                    display_frame = self.draw_boxes(frame.copy(), detections)
                    cv2.imshow('CSI Camera Detection', display_frame)
                    if cv2.waitKey(1) == ord('q'):
                        break

                # 计算FPS
                self.frame_count += 1
                if (time.time() - self.prev_time) > 1:
                    self.fps = self.frame_count / (time.time() - self.prev_time)
                    self.frame_count = 0
                    self.prev_time = time.time()
                    print(f"当前帧率: {self.fps:.1f} FPS")

        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            print("摄像头资源已释放")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='NEUmodel.onnx', help='ONNX模型路径')
    parser.add_argument('--classes', type=str, default='classes.txt', help='类别文件')
    parser.add_argument('--source', type=str, default='tcp://127.0.0.1:8888',
                        help='输入源（0:本地CSI摄像头 / tcp地址）')
    parser.add_argument('--view-img', action='store_true', help='实时显示检测结果')
    args = parser.parse_args()

    detector = CSI_Camera_Detector(args)
    detector.run()