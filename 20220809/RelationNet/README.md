# Relation_Networks

The implementation of Relation Networks for object detection based pytorch.

# Abstract

Although it is well believed for years that modeling relations between objects would help object recognition, there has not been evidence that the idea is working in the deep learning era. All state-of-the-art object detection systems still rely on recognizing object instances individually, without exploiting their relations during learning.

This work proposes an object relation module. It processes a set of objects simultaneously though interaction between their appearance feature and geometry, thus allowing modeling of their relations. It is lightweight and in-place. It does not require additional supervision and is easy to embed in existing networks. It is shown effective on improving object recognition and duplicate removal steps in the modern object detection pipeline. It verifies the efficacy of modeling object relations in CNN based detection. It gives rise to the first fully end-to-end object detector. Code is available at https://github.com/xjtulyc/Relation-Networks-for-Object-Detection

# Train & Test

If you want to train the model, please run the following code.

```
python train.py
```

If you want to test the model, please run the following code.

```
python test.py
```

# Results

We feed the figure into the trained model. The figure is shown bellow and you can get it in the demo fold.

![image](demo/demo.jpg)

And you can get bounding box and detected objects like this.

![image](demo/demo_output.png)

We also compared the relation object detection with Faster RCNN. The detailed experimental data is available in https://github.com/xjtulyc/PKU_Weekly_Summary_repo/blob/main/20220802/Implementation_of_Relation_Networks_for_Object_Detection.pdf