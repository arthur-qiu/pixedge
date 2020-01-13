import torch

def create_model(opt):
    if opt.model == 'pix2pixHD':
        from .pix2pixHD_model import Pix2PixHDModel, InferenceModel
        if opt.isTrain:
            model = Pix2PixHDModel()
        else:
            model = InferenceModel()
    elif opt.model == 'pix2pixHDcls':
        from .pix2pixHDcls_model import Pix2PixHDclsModel, InferenceModel
        if opt.isTrain:
            model = Pix2PixHDclsModel()
        else:
            model = InferenceModel()
    elif opt.model == 'pix2pixHDcls10':
        from .pix2pixHDcls10_model import Pix2PixHDcls10Model, InferenceModel
        if opt.isTrain:
            model = Pix2PixHDcls10Model()
        else:
            model = InferenceModel()
    elif opt.model == 'pix2pixHDedge':
        from .pix2pixHDedge_model import Pix2PixHDedgeModel, InferenceModel
        if opt.isTrain:
            model = Pix2PixHDedgeModel()
        else:
            model = InferenceModel()
    elif opt.model == 'pix2pixHDedgeComb':
        from .pix2pixHDedgeComb_model import Pix2PixHDedgeCombModel, InferenceModel
        if opt.isTrain:
            model = Pix2PixHDedgeCombModel()
        else:
            model = InferenceModel()
    elif opt.model == 'pix2pixHDimgnet':
        from .pix2pixHDimgnet_model import Pix2PixHDimgnetModel, InferenceModel
        if opt.isTrain:
            model = Pix2PixHDimgnetModel()
        else:
            model = InferenceModel()
    elif opt.model == 'pix2pixHDimgnet4':
        from .pix2pixHDimgnet4_model import Pix2PixHDimgnet4Model, InferenceModel
        if opt.isTrain:
            model = Pix2PixHDimgnet4Model()
        else:
            model = InferenceModel()
    else:
    	from .ui_model import UIModel
    	model = UIModel()
    model.initialize(opt)
    if opt.verbose:
        print("model [%s] was created" % (model.name()))

    if opt.isTrain and len(opt.gpu_ids) and not opt.fp16:
        model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)

    return model
