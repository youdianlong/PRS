import torch
import numpy as np
from scipy.ndimage import zoom
from networks.net_factory import net_factory

def load_model(model_name, model_path, num_classes=4):
    print(f"[load_model] Loading model: {model_name} from {model_path}")
    # 强制使用 UNet，无论 model_name 是什么
    net = net_factory(net_type="unet", in_chns=1, class_num=num_classes, mode="train")
    net.load_state_dict(torch.load(model_path))
    net.cuda()
    net.eval()
    return net

def run_inference(model, image_3d):
    prediction = np.zeros_like(image_3d)
    for i in range(image_3d.shape[0]):
        slice = image_3d[i]
        x, y = slice.shape
        resized = zoom(slice, (256/x, 256/y), order=0)
        input_tensor = torch.from_numpy(resized).unsqueeze(0).unsqueeze(0).float().cuda()

        with torch.no_grad():
            out = model(input_tensor)
            if isinstance(out, tuple):
                out = out[0]
            out = torch.softmax(out, dim=1)
            pred = torch.argmax(out, dim=1).squeeze(0).cpu().numpy()
            pred = zoom(pred, (x/256, y/256), order=0)

        prediction[i] = pred
    return prediction
