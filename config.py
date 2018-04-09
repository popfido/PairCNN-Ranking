# coding:utf8
import warnings


class DefaultConfig(object):
    env = 'default'  # visdom 环境
    model = 'PairCNN'  # 使用的模型，名字必须与models/__init__.py中的名字一致

    train_dir = './'
    train_data_root = './data/train/'  # 训练集存放路径
    validate_data_root = './data/validate'  # 验证集存放路径
    test_data_root = './data/test/'  # 测试集存放路径
    load_model_path = None  # 加载预训练的模型的路径，为None代表不加载

    dev_ratio = 0.1  # Ratio of dev/validation data picked from training set
    batch_size = 128  # batch size
    use_gpu = False  # user GPU or not
    num_workers = 4  # how many workers for loading data
    print_freq = 20  # print info every N batch
    eval_freq = 100  # Evaluate model on dev set after this many steps (default: 100)
    checkpoint_freq = 100  # Save model after this many steps (default: 100)

    debug_file = '/tmp/debug'  # if os.path.exists(debug_file): enter ipdb
    result_file = 'result.csv'

    seed = 233  # Random seed (default: 233)
    max_epoch = 20
    lr = 0.1  # initial learning rate
    lr_decay = 0.95  # when val_loss increase, lr = lr*lr_decay

    embedding_dim = 64  # Dimensionality of character embedding (default: 64)
    filter_sizes = "2,3"  # Comma-separated filter sizes (default: '2,3')
    num_filters = 64  # Number of filters per filter size (default: 64)
    num_hidden = 100  # Number of hidden layer units (default: 100)
    dropout_keep_prob = 0.5  # Dropout keep probability (default: 0.5)
    max_len_left = 10  # max document length of left input
    max_len_right = 10  # max document length of right input
    weight_decay = 1e-4  # l2_regularization
    vocab_size = 300000  # Most number of words in vocab (default: 300000)


def parse(self, kwargs):
    """
    根据字典kwargs 更新 config参数
    """
    for k, v in kwargs.items():
        if not hasattr(self, k):
            warnings.warn("Warning: opt has not attribut %s" % k)
        setattr(self, k, v)

    print('user config:')
    for k, v in self.__class__.__dict__.items():
        if not k.startswith('__'):
            print(k, getattr(self, k))


DefaultConfig.parse = parse
opt = DefaultConfig()
# opt.parse = parse
