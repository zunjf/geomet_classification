#Geometry Classification

Thanks to Alan Gray (just copy his work for learning purpose)
- https://agray3.github.io/2016/11/29/Demystifying-Data-Input-to-TensorFlow-for-Deep-Learning.html

I can learning Tensorflow from raw image

Requirement:
- create folder for data (training and validation)
    - Training folder
        - data/train/squares
        - data/train/triangles
    - Validation folder
        - data/validate/squares
        - data/validate/triangles

**command for generate geometry image**

 python build_image_data.py --train_directory=./train --output_directory=./  \
--validation_directory=./validate --labels_file=label.txt   \
--train_shards=1 --validation_shards=1 --num_threads=1 