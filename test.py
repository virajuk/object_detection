from collections import namedtuple

from src.vggderivative import VGGDerivative

deri = VGGDerivative()

# deri.load_data()
# deri.convert_data()
# deri.split_data_train_test()
# deri.write_test_images()

deri.modify_vgg()
