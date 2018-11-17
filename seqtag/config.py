import os


from .general_utils import get_logger
from .data_utils import get_trimmed_glove_vectors, load_vocab, \
        get_processing_word


class Config():
    def __init__(self, data_folder=None, dir_output=None, glove_path=None, load=True, nepochs=15, batch_size=32, dropout=0.2, nepoch_no_imprv=3):
        """Initialize hyperparameters and load vocabs

        Args:
            load_embeddings: (bool) if True, load embeddings into
                np array, else None

        """

        Config.nepochs = nepochs
        Config.batch_size = batch_size
        Config.dropout = dropout
        Config.nepoch_no_imprv = nepoch_no_imprv
        
        Config.dir_output = dir_output
        Config.data_folder = data_folder
        Config.filename_glove = glove_path
        Config.dir_model  = os.path.join(dir_output, "model.weights/")
        Config.path_log   = os.path.join(dir_output, "log.txt")
        Config.filename_trimmed = os.path.join(dir_output, "trimmed_glove.npz")
        Config.filename_train = os.path.join(data_folder, "train.txt")
        Config.filename_dev = os.path.join(data_folder, "valid.txt")
        Config.filename_test = os.path.join(data_folder, "test.txt")  
        
        if not os.path.exists(Config.filename_test):
                Config.filename_test = Config.filename_dev
        
        Config.filename_dev = os.path.join(dir_output, "valid.txt")
        Config.filename_words = os.path.join(dir_output, "words.txt")
        Config.filename_tags = os.path.join(dir_output, "tags.txt")
        Config.filename_chars = os.path.join(dir_output, "chars.txt")

        # directory for training outputs
        if not os.path.exists(self.dir_output):
            os.makedirs(self.dir_output)
        
        # create instance of logger
        self.logger = get_logger(self.path_log)


        


        # load if requested (default)
        if load:
            self.load()


    def load(self):
        """Loads vocabulary, processing functions and embeddings

        Supposes that build_data.py has been run successfully and that
        the corresponding files have been created (vocab and trimmed GloVe
        vectors)

        """
        # 1. vocabulary
        self.vocab_words = load_vocab(self.filename_words)
        self.vocab_tags  = load_vocab(self.filename_tags)
        self.vocab_chars = load_vocab(self.filename_chars)

        self.nwords     = len(self.vocab_words)
        self.nchars     = len(self.vocab_chars)
        self.ntags      = len(self.vocab_tags)

        # 2. get processing functions that map str -> id
        self.processing_word = get_processing_word(self.vocab_words,
                self.vocab_chars, lowercase=True, chars=self.use_chars)
        self.processing_tag  = get_processing_word(self.vocab_tags,
                lowercase=False, allow_unk=False)

        # 3. get pre-trained embeddings
        self.embeddings = (get_trimmed_glove_vectors(self.filename_trimmed)
                if self.use_pretrained else None)


    # general config
    dir_output = None
    dir_model  = None
    path_log   = None

    # embeddings
    dim_word = 300
    dim_char = 100

    # glove files
    filename_glove = None
    # trimmed embeddings (created from glove_filename with build_data.py)
    data_folder = None
    filename_trimmed = None
    use_pretrained = True

    # dataset
    # filename_dev = "data/coNLL/eng/eng.testa.iob"
    # filename_test = "data/coNLL/eng/eng.testb.iob"
    # filename_train = "data/coNLL/eng/eng.train.iob"

    filename_train = None
    filename_dev = None
    filename_test = None
#     filename_dev = filename_test = filename_train = "data/test.txt" # test

    max_iter = None # if not None, max number of examples in Dataset

    # vocab (created from dataset with build_data.py)
    filename_words = None
    filename_tags = None
    filename_chars = None

    # training
    train_embeddings = False
    nepochs          = None
    dropout          = None
    batch_size       = None
    lr_method        = "adam"
    lr               = 0.001
    lr_decay         = 0.9
    clip             = -1 # if negative, no clipping
    nepoch_no_imprv  = None

    # model hyperparameters
    hidden_size_char = 100 # lstm on chars
    hidden_size_lstm = 300 # lstm on word embeddings

    # NOTE: if both chars and crf, only 1.6x slower on GPU
    use_crf = True # if crf, training is 1.7x slower on CPU
    use_chars = True # if char embedding, training is 3.5x slower on CPU
