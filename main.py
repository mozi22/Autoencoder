import tensorflow as tf
import input_pipeline as inpp



class Autoencoder:

    def __init__(self):
        print('mozi')
        self.dataset = inpp.parse()
        self.iterator = self.dataset.make_initializable_iterator()
        rainyImage = self.iterator.get_next()

        print(rainyImage)


obj = Autoencoder()