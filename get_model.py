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
                      'deeplabv3plus', 'fcn', 'unet', 'MFIANet_es', 'MFIANet_em', 'MFIANet_el', 'MFIANet_b0',
                      'MFIANet_b1', 'MFIANet_b2', 'MFIANet_b3', 'MFIANet_b4', 'MFIANet_b5']

    if models == 'MFIANet_b0':
        from model.backbone.MFIANet_mit import MFIANet_b0
        model = MFIANet_b0(num_classes=args.num_classes, in_chans=args.data_channel, aux=args.use_aux, head=args.head,
                           head_type=args.head_type).cuda()

    if models == 'MFIANet_b1':
        from model.backbone.MFIANet_mit import MFIANet_b1
        model = MFIANet_b1(num_classes=args.num_classes, in_chans=args.data_channel, aux=args.use_aux, head=args.head,
                           head_type=args.head_type).cuda()

    if models == 'MFIANet_b2':
        from model.backbone.MFIANet_mit import MFIANet_b2
        model = MFIANet_b2(num_classes=args.num_classes, in_chans=args.data_channel, aux=args.use_aux, head=args.head,
                           head_type=args.head_type).cuda()

    if models == 'MFIANet_b3':
        from model.backbone.MFIANet_mit import MFIANet_b3
        model = MFIANet_b3(num_classes=args.num_classes, in_chans=args.data_channel, aux=args.use_aux, head=args.head,
                           head_type=args.head_type).cuda()

    if models == 'MFIANet_b4':
        from model.backbone.MFIANet_mit import MFIANet_b4
        model = MFIANet_b4(num_classes=args.num_classes, in_chans=args.data_channel, aux=args.use_aux, head=args.head,
                           head_type=args.head_type).cuda()

    if models == 'MFIANet_b5':
        from model.backbone.MFIANet_mit import MFIANet_b5
        model = MFIANet_b5(num_classes=args.num_classes, in_chans=args.data_channel, aux=args.use_aux, head=args.head,
                           head_type=args.head_type).cuda()

    if models == 'MFIANet_es':
        from model.backbone.MFIANet_efficientnet import MFIANet_s
        model = MFIANet_s(num_classes=args.num_classes, in_chans=args.data_channel, aux=args.use_aux, head=args.head,
                          head_type=args.head_type).cuda()

    if models == 'MFIANet_em':
        from model.backbone.MFIANet_efficientnet import MFIANet_m
        model = MFIANet_m(num_classes=args.num_classes, in_chans=args.data_channel, aux=args.use_aux, head=args.head,
                          head_type=args.head_type).cuda()

    if models == 'MFIANet_el':
        from model.backbone.MFIANet_efficientnet import MFIANet_l
        model = MFIANet_l(num_classes=args.num_classes, in_chans=args.data_channel, aux=args.use_aux, head=args.head,
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