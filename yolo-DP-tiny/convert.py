"""
convert_to_deploy.py
Chuyển đổi model đã train (MobileOneBlock training mode)
→ deploy mode (1 conv duy nhất, inference nhanh hơn)

Cách dùng:
    python convert_to_deploy.py \
        --weights runs/train/exp/weights/best.pt \
        --output  weights/best_deploy.pt

    # Hoặc chỉ định device:
    python convert_to_deploy.py --weights best.pt --output best_deploy.pt --device cpu
"""

import argparse
import copy
import torch
from pathlib import Path


# ── Import các module cần thiết từ project ──────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent))   # thêm thư mục gốc vào path

from models.common import MobileOneBlock, DSConv_MO   # adjust nếu path khác


# ════════════════════════════════════════════════════════════════════════════
def switch_model_to_deploy(model):
    """
    Duyệt toàn bộ model, gọi switch_to_deploy() trên mọi
    MobileOneBlock đang ở training mode.

    Returns:
        model đã chuyển sang deploy (in-place),
        counts: số block đã convert
    """
    count = 0
    for name, module in model.named_modules():
        if isinstance(module, MobileOneBlock):
            if not module.deploy:
                module.switch_to_deploy()
                count += 1
    return model, count


def load_model(weights_path, device):
    """Load checkpoint YOLOv5, trả về model và metadata."""
    ckpt = torch.load(weights_path, map_location=device)

    # YOLOv5 lưu model dưới key 'model' hoặc trực tiếp
    if isinstance(ckpt, dict):
        model = ckpt.get('model', ckpt.get('ema', None))
        if model is None:
            raise KeyError("Không tìm thấy key 'model' hay 'ema' trong checkpoint.")
    else:
        model = ckpt
        ckpt  = {'model': model}   # wrap lại để lưu đồng nhất

    model = model.float().eval()
    return model, ckpt


def verify_output(model_train, model_deploy, device, img_size=640):
    """
    Kiểm tra output của train vs deploy mode có xấp xỉ nhau không.
    Sai số cho phép: < 1e-4 (float32 rounding).
    """
    dummy = torch.zeros(1, 3, img_size, img_size).to(device)

    with torch.no_grad():
        out_train  = model_train(dummy)
        out_deploy = model_deploy(dummy)

    # Lấy tensor đầu tiên để so sánh
    if isinstance(out_train, (tuple, list)):
        out_train  = out_train[0]
        out_deploy = out_deploy[0]

    diff = (out_train - out_deploy).abs().max().item()
    return diff


# ════════════════════════════════════════════════════════════════════════════
def main(args):
    device = torch.device(args.device)
    print(f"\n{'='*55}")
    print(f"  Convert MobileOne → Deploy Mode")
    print(f"{'='*55}")
    print(f"  Weights : {args.weights}")
    print(f"  Output  : {args.output}")
    print(f"  Device  : {device}\n")

    # 1. Load model gốc (training mode)
    print("[1/4] Loading checkpoint...")
    model_train, ckpt = load_model(args.weights, device)
    model_train = model_train.to(device)

    # Đếm số MobileOneBlock trước khi convert
    total_blocks = sum(
        1 for m in model_train.modules()
        if isinstance(m, MobileOneBlock)
    )
    print(f"      Tìm thấy {total_blocks} MobileOneBlock trong model")

    # 2. Deep copy để giữ bản gốc (dùng để verify)
    if args.verify:
        print("[2/4] Tạo bản copy để verify...")
        model_orig = copy.deepcopy(model_train).eval()
    else:
        print("[2/4] Bỏ qua verify (--no-verify)")
        model_orig = None

    # 3. Convert sang deploy
    print("[3/4] Chuyển đổi sang deploy mode...")
    model_deploy, converted = switch_model_to_deploy(model_train)
    model_deploy.eval()

    # Tắt gradient toàn bộ model — không cần thiết cho inference
    for param in model_deploy.parameters():
        param.requires_grad = False

    # Kiểm tra xác nhận
    still_grad = sum(1 for p in model_deploy.parameters() if p.requires_grad)
    if still_grad == 0:
        print(f"      ✓ Đã convert {converted}/{total_blocks} blocks")
        print(f"      ✓ Gradient đã tắt hoàn toàn (requires_grad=False)")
    else:
        print(f"      ⚠ Vẫn còn {still_grad} parameter có gradient!")

    # 4. Verify (so sánh output)
    if args.verify and model_orig is not None:
        print("[4/4] Verifying output...")
        try:
            diff = verify_output(model_orig, model_deploy, device, args.imgsz)
            status = "✓ PASS" if diff < 1e-3 else "⚠ WARNING"
            print(f"      {status}  max diff = {diff:.2e}  "
                  f"(ngưỡng: 1e-3)")
            if diff >= 1e-3:
                print("      ⚠ Sai số lớn hơn dự kiến, kiểm tra lại")
        except Exception as e:
            print(f"      ⚠ Verify thất bại: {e}")
    else:
        print("[4/4] Bỏ qua verify")

    # 5. Lưu checkpoint mới
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Giữ nguyên metadata của checkpoint gốc, chỉ thay model
    save_ckpt = {
        **{k: v for k, v in ckpt.items() if k != 'model'},
        'model': model_deploy,
        'deploy': True,           # flag để nhận biết sau này
    }
    torch.save(save_ckpt, output_path)

    # Kiểm tra kích thước file
    size_before = Path(args.weights).stat().st_size / 1e6
    size_after  = output_path.stat().st_size / 1e6
    print(f"\n  ✓ Đã lưu: {output_path}")
    print(f"  Kích thước: {size_before:.1f} MB → {size_after:.1f} MB")
    print(f"\n  Dùng file này để export ONNX / TFLite / TensorRT\n")


# ════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert MobileOne training weights → deploy mode'
    )
    parser.add_argument(
        '--weights', type=str, required=True,
        help='Path đến file .pt đã train  (vd: runs/train/exp/weights/best.pt)'
    )
    parser.add_argument(
        '--output', type=str, default='weights/best_deploy.pt',
        help='Path để lưu file deploy  (default: weights/best_deploy.pt)'
    )
    parser.add_argument(
        '--device', type=str, default='cpu',
        help='cpu hoặc cuda:0  (default: cpu)'
    )
    parser.add_argument(
        '--imgsz', type=int, default=640,
        help='Image size để verify  (default: 640)'
    )
    parser.add_argument(
        '--no-verify', dest='verify', action='store_false',
        help='Bỏ qua bước verify output'
    )
    parser.set_defaults(verify=True)

    args = parser.parse_args()
    main(args)