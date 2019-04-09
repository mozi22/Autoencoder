import tensorflow as tf

def _parse_function(filename):
  r_image_string = tf.read_file(filename)
  r_image_decoded = tf.image.decode_jpeg(r_image_string,channels=3)
  r_image_decoded = tf.reshape(r_image_decoded,[256,256,3])
  r_image_decoded = tf.image.convert_image_dtype(r_image_decoded,tf.float32)

  # r_image_decoded = tf.divide(r_image_decoded,[255])


  r_image_decoded_std = tf.image.resize_images(r_image_decoded,[128,128])
  r_image_decoded_std = tf.image.per_image_standardization(r_image_decoded)


  return r_image_decoded_std



def parse():
  # rainy_start = 637
  # rainy_end = 1207
  rainy_start = 1
  rainy_end = 607

  rainy_files = []

  for i in range(rainy_start,rainy_end):
    file_id = "{0:0=4d}".format(i)
    rainy_files.append('./sunny/000'+str(file_id)+'.jpeg')


  rainy_filenames = tf.constant(rainy_files)

  dataset = tf.data.Dataset.from_tensor_slices(rainy_filenames)
  dataset = dataset.map(_parse_function).repeat().shuffle(buffer_size=50).batch(4,True)

  return dataset
