
class params():
    def __init__(self):
        "hrnet48"
        self.STAGE2 = {'NUM_MODULES': 1,
                        'NUM_BRANCHES': 2,
                        'NUM_BLOCKS': [4,4],
                        'NUM_CHANNELS': [48,96],
                        'BLOCK':'BASIC',
                        'FUSE_METHOD': 'SUM'}
        self.STAGE3 = {'NUM_MODULES': 4,
                       'NUM_BRANCHES': 3,
                       'NUM_BLOCKS': [6, 6, 6],
                       'NUM_CHANNELS': [48, 96, 192],
                       'BLOCK': 'BASIC',
                       'FUSE_METHOD': 'SUM'}
        self.STAGE4 = {'NUM_MODULES': 3,
                       'NUM_BRANCHES': 4,
                       'NUM_BLOCKS': [3, 3, 3, 3],
                       'NUM_CHANNELS': [48, 96, 192, 384],
                       'BLOCK': 'BASIC',
                       'FUSE_METHOD': 'SUM'}
        # "hrnet32"
        # self.STAGE2 = {'NUM_MODULES': 1,
        #                'NUM_BRANCHES': 2,
        #                'NUM_BLOCKS': [4, 4],
        #                'NUM_CHANNELS': [32, 64],
        #                'BLOCK': 'BASIC',
        #                'FUSE_METHOD': 'SUM'}
        # self.STAGE3 = {'NUM_MODULES': 4,
        #                'NUM_BRANCHES': 3,
        #                'NUM_BLOCKS': [4, 4, 4],
        #                'NUM_CHANNELS': [32, 64, 128],
        #                'BLOCK': 'BASIC',
        #                'FUSE_METHOD': 'SUM'}
        # self.STAGE4 = {'NUM_MODULES': 3,
        #                'NUM_BRANCHES': 4,
        #                'NUM_BLOCKS': [4, 4, 4, 4],
        #                'NUM_CHANNELS': [32, 64, 128, 256],
        #                'BLOCK': 'BASIC',
        #                'FUSE_METHOD': 'SUM'}

        # "hrnet32_light"
        # self.STAGE2 = {'NUM_MODULES': 1,
        #                'NUM_BRANCHES': 2,
        #                'NUM_BLOCKS': [4, 4],
        #                'NUM_CHANNELS': [32, 64],
        #                'BLOCK': 'BASIC',
        #                'FUSE_METHOD': 'SUM'}
        # self.STAGE3 = {'NUM_MODULES': 1,
        #                'NUM_BRANCHES': 3,
        #                'NUM_BLOCKS': [6, 6, 6],
        #                'NUM_CHANNELS': [32, 64, 128],
        #                'BLOCK': 'BASIC',
        #                'FUSE_METHOD': 'SUM'}
        # self.STAGE4 = {'NUM_MODULES': 1,
        #                'NUM_BRANCHES': 4,
        #                'NUM_BLOCKS': [3, 3, 3, 3],
        #                'NUM_CHANNELS': [32, 64, 128, 256],
        #                'BLOCK': 'BASIC',
        #                'FUSE_METHOD': 'SUM'}
