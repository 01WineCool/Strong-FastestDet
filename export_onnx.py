import os
import argparse
import torch
from module.detector import Detector
from FastestDet.utils.tool import LoadYaml


def export_onnx(cfg_path, weight_path, output_path="output.onnx", opset=13):
    """PyTorch模型转ONNX工具（支持动态输入优化）"""
    # ----------------------- 参数校验 -----------------------
    assert os.path.exists(cfg_path), f"配置文件 {cfg_path} 不存在"
    assert os.path.exists(weight_path), f"权重文件 {weight_path} 不存在"

    # ----------------------- 模型加载 -----------------------
    cfg = LoadYaml(cfg_path)
    model = Detector(cfg.category_num, load_param=True)

    # 加载权重（过滤无关键值）
    state_dict = torch.load(weight_path, map_location='cpu')
    filtered_dict = {k: v for k, v in state_dict.items()
                     if not k.endswith(('total_ops', 'total_params'))}
    model.load_state_dict(filtered_dict)
    model.eval()

    # ----------------------- 动态输入配置 -----------------------
    dummy_input = torch.randn(1, 3, cfg.input_height, cfg.input_width)

    # 修正动态轴设置（仅保留batch维度动态）
    dynamic_axes = {
        'images': {0: 'batch'},  # 仅batch维度动态
        'output': {0: 'batch'}  # 输出保持对应batch
    }

    # ----------------------- ONNX导出 -----------------------
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        verbose=True,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=['images'],
        output_names=['output'],
        dynamic_axes=dynamic_axes,
        training=torch.onnx.TrainingMode.EVAL,
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK
    )

    # ----------------------- 模型优化 -----------------------
    try:
        import onnx
        from onnxsim import simplify
        from onnx import shape_inference

        print("\n正在优化ONNX模型...")

        # 执行形状推断
        onnx_model = onnx.load(output_path)
        onnx_model = shape_inference.infer_shapes(onnx_model)

        # 显式指定输入形状
        model_simp, check = simplify(
            onnx_model,
            test_input_shapes={'images': [1, 3, cfg.input_height, cfg.input_width]},
            perform_optimization=True
        )
        assert check, "优化失败：输出不匹配"

        # 保存优化后模型
        onnx.save(model_simp, output_path)
    except Exception as e:
        print(f"⚠️ 优化失败：{str(e)}")
        # 保存原始模型以供调试
        onnx.save(onnx_model, output_path.replace(".onnx", "_raw.onnx"))
        raise

    print(f"✅ ONNX模型已保存至：{os.path.abspath(output_path)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch模型转ONNX工具')
    parser.add_argument('--yaml', type=str,
                        default=r"C:\\Users\\Wu Meishun\\Desktop\\Python\\FastestDet\\configs\\coco.yaml",
                        help='配置文件路径')
    parser.add_argument('--weight', type=str,
                        default="checkpoint/weight_AP05_0.4122_270-epoch.pth",
                        help='权重文件路径')
    parser.add_argument('--output', type=str,
                        default='NEUmodel.onnx',
                        help='输出路径')
    parser.add_argument('--opset', type=int,
                        default=13,
                        help='ONNX算子集版本（建议12-15）')
    args = parser.parse_args()

    export_onnx(
        cfg_path=args.yaml,
        weight_path=args.weight,
        output_path=args.output,
        opset=args.opset
    )


"""
后续的运行
python3 detect.py 
    --model NEUmodel.onnx 
    --classes .configs/coco.names.txt 
    --source tcp://127.0.0.1:8888 
    --view-img
"""