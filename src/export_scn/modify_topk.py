import numpy as np
import onnx
import onnx_graphsurgeon as gs
from onnxsim import simplify


def modify_topk(onnx_file: str, sim: bool):
    model = onnx.load(onnx_file)
    if sim:
        model, check = simplify(model)
        assert check, "Simplified ONNX model could not be validated"
    graph = gs.import_onnx(model)
    nodes = [node for node in graph.nodes if node.op == "TopK"]
    if not nodes:
        return
    topk = nodes[0]
    k = graph.outputs[0].shape[2]
    topk.inputs[1] = gs.Constant("K", values=np.array([k], dtype=np.int64))
    topk.outputs[0].shape = [1, k]
    topk.outputs[0].dtype = topk.inputs[0].dtype if topk.inputs[0].dtype else np.float32
    topk.outputs[1].shape = [1, k]
    topk.outputs[1].dtype = np.int64
    graph.cleanup().toposort()
    onnx.save_model(gs.export_onnx(graph), onnx_file)
