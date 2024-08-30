import model.config.vit_seg_configs as configs
from model.config.hrcnet_configs import params

def get_model(args, models='danet'):
    if models in ['segformer_b0', 'segformer_b1', 'segformer_b2',  'segformer_b3',  'segformer_b4', 'segformer_b5',
                  'efficientnetv2', 'cswinT', 'cswinS', 'cswin_B', 'cswin_L']:
        print(models, args.head_type)
    else:
        print(models)

    assert models in ['transunet', 'hrcnet', 'segformer_b0', 'segformer_b1', 'segformer_b2', 'segformer_b3',
                      'segformer_b4', 'segformer_b5', 'cswinT', 'cswinS', 'cswinB', 'cswinL', 'efficientnetv2',
                      'deeplabv3plus', 'fcn', 'unet', 'GACNet_es', 'GACNet_em', 'GACNet_el', 'GACNet_b0',
                      'GACNet_b1', 'GACNet_b2', 'GACNet_b3', 'GACNet_b4', 'GACNet_b5']

    if models == 'GACNet_b0':
        from model.backbone.MFIANet_mit import MFIANet_b0
        model = GACNet_b0(num_classes=args.num_classes, in_chans=args.data_channel, aux=args.use_aux, head=args.head,
                           head_type=args.head_type).cuda()

    if models == 'GACNet_b1':
        from model.backbone.GACNet_mit import GACNet_b1
        model = GACNet_b1(num_classes=args.num_classes, in_chans=args.data_channel, aux=args.use_aux, head=args.head,
                           head_type=args.head_type).cuda()

    if models == 'GACNet_b2':
        from model.backbone.GACNet_mit import GACNet_b2
        model = MFIANet_b2(num_classes=args.num_classes, in_chans=args.data_channel, aux=args.use_aux, head=args.head,
                           head_type=args.head_type).cuda()

    if models == 'GACNet_b3':
        from model.backbone.GACNet_mit import GACNet_b3
        model = GACNet_b3(num_classes=args.num_classes, in_chans=args.data_channel, aux=args.use_aux, head=args.head,
                           head_type=args.head_type).cuda()

    if models == 'GACNet_b4':
        from model.backbone.GACNet_mit import GACNet_b4
        model = GACNet_b4(num_classes=args.num_classes, in_chans=args.data_channel, aux=args.use_aux, head=args.head,
                           head_type=args.head_type).cuda()

    if models == 'GACNet_b5':
        from model.backbone.GACNet_mit import GACNet_b5
        model = GACNet_b5(num_classes=args.num_classes, in_chans=args.data_channel, aux=args.use_aux, head=args.head,
                           head_type=args.head_type).cuda()

    if models == 'GACNet_es':
        from model.backbone.GACNet_efficientnet import GACNet_s
        model = GACNet_s(num_classes=args.num_classes, in_chans=args.data_channel, aux=args.use_aux, head=args.head,
                          head_type=args.head_type).cuda()

    if models == 'GACNet_em':
        from model.backbone.GACNet_efficientnet import GACNet_m
        model = GACNet_m(num_classes=args.num_classes, in_chans=args.data_channel, aux=args.use_aux, head=args.head,
                          head_type=args.head_type).cuda()

    if models == 'GACNet_el':
        from model.backbone.GACNet_efficientnet import GACNet_l
        model = GACNet_l(num_classes=args.num_classes, in_chans=args.data_channel, aux=args.use_aux, head=args.head,
                          head_type=args.head_type).cuda()

    if models == 'deeplabv3plus':
        from model.backbone.deeplabv3_plus import DeepLab
        model = DeepLab(num_classes=args.num_classes, in_chans=args.data_channel, pretrained=False).cuda()

    if models == 'fcn':
        from model.backbone.fcn import FCN16s
        model = FCN16s(nclass=args.num_classes, in_chans=args.data_channel, aux=args.use_aux).cuda()

    if models == 'unet':
        from model.backbone.unet import UNet
        model = UNet(nclass=args.num_classes, in_chans=args.data_channel).cuda()


    if models == 'efficientnetv2':
        from model.backbone.efficientnetv2 import efficientnetv2_s as efficient
        model = efficient(num_classes=args.num_classes, in_chans=args.data_channel, aux=args.use_aux,
                              head=args.head_type).cuda()

    if models == 'cswinT':
        from model.backbone.cswinT import cswin_tiny as cswin
        model = cswin(nclass=args.num_classes, img_size=args.img_size, in_chans=args.data_channel, pretrained=False,
                          aux=args.use_aux, head=args.head_type).cuda()

    if models == 'cswinS':
        from model.backbone.cswinT import cswin_small as cswin
        model = cswin(nclass=args.num_classes, img_size=args.img_size, in_chans=args.data_channel, pretrained=False,
                          aux=args.use_aux, head=args.head_type).cuda()

    if models == 'cswinB':
        from model.backbone.cswinT import cswin_base as cswin
        model = cswin(nclass=args.num_classes, img_size=args.img_size, in_chans=args.data_channel, pretrained=False,
                          aux=args.use_aux, head=args.head_type).cuda()

    if models == 'cswinL':
        from model.backbone.cswinT import cswin_large as cswin
        model = cswin(nclass=args.num_classes, img_size=args.img_size, in_chans=args.data_channel, pretrained=False,
                          aux=args.use_aux, head=args.head_type).cuda()

    if models == 'transunet':
        from model.backbone.transunet import VisionTransformer as TransUNet
        model = TransUNet(configs.get_r50_b16_config(), img_size=args.img_size, num_classes=args.num_classes,
                              in_chans=args.data_channel, pretrained=False).cuda()

    if models == 'hrcnet':
        from model.backbone.hrcnet import PoseHighResolutionNet as HRCNet
        model = HRCNet(params(), num_class=args.num_classes, in_chans=args.data_channel).cuda()

    if models == 'segformer_b0':
        from  model.backbone.segformer import segformer_b0 as SegFormer_B0
        model = SegFormer_B0(num_classes=args.num_classes, in_chans=args.data_channel, aux=args.use_aux, pretrained=False,
                                 head=args.head_type)

    if models == 'segformer_b1':
        from  model.backbone.segformer import segformer_b1 as SegFormer_B1
        model = SegFormer_B1(num_classes=args.num_classes, in_chans=args.data_channel, aux=args.use_aux, pretrained=False,
                                 head=args.head_type)

    if models == 'segformer_b2':
        from model.backbone.segformer import segformer_b2 as SegFormer_B2
        model = SegFormer_B2(num_classes=args.num_classes, in_chans=args.data_channel, aux=args.use_aux, pretrained=False,
                                 head=args.head_type).cuda()

    if models == 'segformer_b3':
        from model.backbone.segformer import segformer_b3 as SegFormer_B3
        model = SegFormer_B3(num_classes=args.num_classes, in_chans=args.data_channel, aux=args.use_aux, pretrained=False,
                                 head=args.head_type).cuda()

    if models == 'segformer_b4':
        from model.backbone.segformer import segformer_b4 as SegFormer_B4
        model = SegFormer_B4(num_classes=args.num_classes, in_chans=args.data_channel, aux=args.use_aux, pretrained=False,
                                 head=args.head_type).cuda()

    if models == 'segformer_b5':
        from  model.backbone.segformer import segformer_b5 as SegFormer_B5
        model = SegFormer_B5(num_classes=args.num_classes, in_chans=args.data_channel, aux=args.use_aux, pretrained=False,
                                 head=args.head_type)

    return model
