from types import MethodType

import click
import torch
from mmdet3d.apis import init_model

from .modify_topk import modify_topk
from .exptool import export_onnx
from .funcs import layer_fusion_bn_relu


def _forward(self, x):
    x = self.pts_backbone(x)
    x = self.pts_neck(x)
    x = (
        self.pts_bbox_head(x)
        if hasattr(self, "pts_bbox_head")
        else self.bbox_head(x, metas={})
    )
    return x


@click.command()
@click.argument("config")
@click.argument("ckpt")
@click.option("--input", default="deploy/data/input.pth")
@click.option("--sim", is_flag=True, default=False)
@click.option("--in_channel", default=5)
def main(config, ckpt, input, sim, in_channel):
    model = init_model(config, ckpt)
    model.eval().cuda()
    model.forward = MethodType(_forward, model)
    if hasattr(model, "pts_middle_encoder"):
        voxels = torch.zeros(1, in_channel).cuda().half()
        coors  = torch.zeros(1, 4).int().cuda()
        batch_size = 1
        model.pts_middle_encoder = layer_fusion_bn_relu(model.pts_middle_encoder)
        for _, module in model.named_modules():
            module.precision = "fp16"
            module.output_precision = "fp16"
        model.pts_middle_encoder.conv_input.precision = "fp16"
        model.pts_middle_encoder.conv_out.output_precision = "fp16"
        model_name = type(model).__name__.lower()
        export_onnx(model.pts_middle_encoder.half(), voxels, coors, batch_size, False, f"{model_name}_pts_middle_encoder.onnx")
        # out = model.pts_middle_encoder.forward(voxels, coors, batch_size)
        torch.onnx.export(
            model,
            torch.randn(1, 256, 180, 180).cuda(),
            f"{model_name}_pts_backbone_neck_head.onnx",
            input_names=["intput"],
            # output_names=["output0", "output1"],
            opset_version=17,
        )

        modify_topk(f"{model_name}_pts_backbone_neck_head.onnx", sim)
    else:
        print("model doesn't have pts middle encoder")


if __name__ == "__main__":
    main()
