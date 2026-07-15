# ════════════════════════════════════════════════════════════════════════════
# [1]  MobileOneBlock  (Apple CVPR 2023 — arxiv 2206.04040)
#      Thay thế ACBlock trong DSConv_AC
#
#  Train: k nhánh 3×3 + k nhánh 1×1 + 1 identity (khi c1==c2)
#  Infer: merge thành 1 conv 3×3 duy nhất  →  zero overhead
#
#  So với ACBlock cũ (3×3 + 1×3 + 3×1):
#    - Đa dạng hơn: k nhân đôi conv giúp over-param mạnh hơn
#    - Có thể scale k (default k=4) theo tài nguyên
# ════════════════════════════════════════════════════════════════════════════
class MobileOneBlock(nn.Module):
    """
    MobileOneBlock: multi-branch re-parameterizable conv.

    Args:
        c1      : input channels
        c2      : output channels
        k       : number of parallel 3×3 branches (default 4)
        s       : stride
        g       : groups  (g=c1 → depthwise, dùng trong DSConv_MO)
        deploy  : True khi inference/export → dùng 1 conv duy nhất
    """

    def __init__(self, c1, c2, k=4, s=1, g=1, act=True, deploy=False):
        super().__init__()
        self.deploy = deploy
        self.g = g
        self.c2 = c2
        self.act = Conv.default_act if act is True else (
            act if isinstance(act, nn.Module) else nn.Identity())

        if deploy:
            # ── Inference mode: 1 conv 3×3 ──────────────────────────────
            self.reparam = nn.Conv2d(c1, c2, 3, s, 1, groups=g, bias=True)
        else:
            # ── Training: k nhánh 3×3, k nhánh 1×1, (optional) identity ─
            pad = autopad(3, None)

            # k branches 3×3
            self.branches_3x3 = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(c1, c2, 3, s, pad, groups=g, bias=False),
                    nn.BatchNorm2d(c2)
                ) for _ in range(k)
            ])

            # k branches 1×1  (chỉ hợp lệ khi stride=1)
            if s == 1:
                self.branches_1x1 = nn.ModuleList([
                    nn.Sequential(
                        nn.Conv2d(c1, c2, 1, 1, 0, groups=g, bias=False),
                        nn.BatchNorm2d(c2)
                    ) for _ in range(k)
                ])
            else:
                self.branches_1x1 = None

            # Identity branch (chỉ khi c1==c2 và stride=1)
            if c1 == c2 and s == 1:
                self.bn_identity = nn.BatchNorm2d(c2)
            else:
                self.bn_identity = None

    def forward(self, x):
        if self.deploy:
            return self.act(self.reparam(x))

        out = sum(b(x) for b in self.branches_3x3)

        if self.branches_1x1 is not None:
            out = out + sum(b(x) for b in self.branches_1x1)

        if self.bn_identity is not None:
            out = out + self.bn_identity(x)

        return self.act(out)

    # ── Helper: fold BN vào conv ─────────────────────────────────────────
    @staticmethod
    def _fuse_conv_bn(conv, bn):
        w = conv.weight
        std = (bn.running_var + bn.eps).sqrt()
        scale = (bn.weight / std).reshape(-1, 1, 1, 1)
        fused_w = w * scale
        fused_b = bn.bias - bn.running_mean * bn.weight / std
        return fused_w, fused_b

    def switch_to_deploy(self):
        """Gọi trước khi inference hoặc export ONNX/TFLite."""
        if self.deploy:
            return

        # Tổng hợp weight từ tất cả nhánh 3×3
        K, B = 0.0, 0.0

        for seq in self.branches_3x3:
            w, b = self._fuse_conv_bn(seq[0], seq[1])
            K = K + w
            B = B + b

        # Pad 1×1 → 3×3 rồi cộng vào
        if self.branches_1x1 is not None:
            for seq in self.branches_1x1:
                w, b = self._fuse_conv_bn(seq[0], seq[1])
                K = K + F.pad(w, [1, 1, 1, 1])
                B = B + b

        # Identity branch: BN(x)  =  BN(I * x)  →  pad identity kernel
        if self.bn_identity is not None:
            bn = self.bn_identity
            c = self.c2 // self.g
            # identity conv weight cho một group
            id_w = torch.zeros(self.c2, c, 3, 3,
                               device=bn.weight.device, dtype=bn.weight.dtype)
            for i in range(self.c2):
                id_w[i, i % c, 1, 1] = 1.0
            std = (bn.running_var + bn.eps).sqrt()
            scale = (bn.weight / std).reshape(-1, 1, 1, 1)
            K = K + id_w * scale
            B = B + bn.bias - bn.running_mean * bn.weight / std

        # Gán vào conv deploy
        self.reparam = nn.Conv2d(
            self.branches_3x3[0][0].in_channels,
            self.c2, 3,
            self.branches_3x3[0][0].stride,
            1, groups=self.g, bias=True
        )
        self.reparam.weight.data = K
        self.reparam.bias.data = B

        # Xóa nhánh training
        for attr in ['branches_3x3', 'branches_1x1', 'bn_identity']:
            if hasattr(self, attr) and getattr(self, attr) is not None:
                delattr(self, attr)
        self.deploy = True


class DSConv_MO(nn.Module):
    """
    DSConv cải tiến: thay ACBlock bằng MobileOneBlock.
    DW (depthwise MobileOne) + PW (LowRankConv1x1 giữ nguyên)

    Dùng để thay thế DSConv_AC trong yaml:
        DSConv_AC  →  DSConv_MO
    """

    def __init__(self, c1, c2, k_branches=4, s=1, r=None, act=True, deploy=False):
        super().__init__()
        self.dw = MobileOneBlock(c1, c1, k=k_branches, s=s,
                                 g=c1, act=False, deploy=deploy)
        self.pw = LowRankConv1x1(c1, c2, r=r, act=act)

    def forward(self, x):
        return self.pw(self.dw(x))

    def switch_to_deploy(self):
        self.dw.switch_to_deploy()


# ════════════════════════════════════════════════════════════════════════════
# [2]  PConv + FasterNet Block  (CVPR 2023 — arxiv 2303.03667)
#      Thay thế DWConv trong bottleneck và DSConv
#
#  Ý tưởng: feature maps có NHIỀU redundancy → chỉ cần conv trên 1/n_div channels
#  Phần còn lại giữ nguyên (zero cost) → FLOPs giảm × n_div, memory access giảm
#  Thực tế NHANH HƠN DWConv trên CPU/ARM vì cache-friendly
# ════════════════════════════════════════════════════════════════════════════
class PConv(nn.Module):
    """
    Partial Convolution (PConv) từ FasterNet (CVPR 2023).

    Chỉ apply conv 3×3 lên (c1 // n_div) channels đầu,
    phần còn lại copy qua không đổi.

    Args:
        c1      : input/output channels (PConv giữ channels)
        k       : kernel size (mặc định 3)
        n_div   : phân chia ratio — conv trên 1/n_div channels (mặc định 4)
        forward : 'split_cat' (nhanh hơn) hoặc 'slicing'
    """

    def __init__(self, c1, k=3, n_div=4, forward='split_cat'):
        super().__init__()
        self.dim_conv = c1 // n_div
        self.dim_untouched = c1 - self.dim_conv
        self.conv = nn.Conv2d(
            self.dim_conv, self.dim_conv,
            k, 1, k // 2, bias=False
        )

        if forward == 'slicing':
            self.forward = self.forward_slicing
        else:
            self.forward = self.forward_split_cat

    def forward_slicing(self, x):
        x = x.clone()
        x[:, :self.dim_conv, :, :] = self.conv(x[:, :self.dim_conv, :, :])
        return x

    def forward_split_cat(self, x):
        x1, x2 = torch.split(x, [self.dim_conv, self.dim_untouched], dim=1)
        x1 = self.conv(x1)
        return torch.cat([x1, x2], dim=1)


class FasterNetBlock(nn.Module):
    """
    FasterNet Block: PConv + PWConv (pointwise) + activation.
    Giữ channels (c1 == c2), dùng như C2f bottleneck.

    Cấu trúc: PConv → BN+Act (middle) → PWConv → BN
    (Activation chỉ sau PConv để giảm memory)
    """

    def __init__(self, c1, n_div=4, act=True):
        super().__init__()
        self.pconv = PConv(c1, k=3, n_div=n_div)
        self.bn1 = nn.BatchNorm2d(c1)
        self.act = Conv.default_act if act is True else nn.Identity()
        self.pwconv = nn.Conv2d(c1, c1, 1, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(c1)

    def forward(self, x):
        shortcut = x
        x = self.act(self.bn1(self.pconv(x)))
        x = self.bn2(self.pwconv(x))
        return x + shortcut  # residual


class C2f_Faster(nn.Module):
    """
    C2f với FasterNet blocks bên trong — thay C2f_LR.

    Dùng thay thế C2f_LR trong yaml:
        C2f_LR  →  C2f_Faster
    """

    def __init__(self, c1, c2, n=1, shortcut=False, n_div=4, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1, 1)
        self.m = nn.ModuleList(
            FasterNetBlock(self.c, n_div=n_div)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))