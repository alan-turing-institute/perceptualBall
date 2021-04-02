# PerceptualBall

This is the code for the paper: 
[Explaining Classifiers using Adversarial Perturbations on the Perceptual Ball](https://arxiv.org/abs/1912.09405) by Andrew Elliott, Stephen Law and Chris Russell. 

We provide an implementation of the saliency method in `common_code.py` which can be integrated into other pipelines, as well as some additional routines which are useful for plotting etc. We provide an example of this code in `example.py`. 

In addition, we provide a jupyter notebook demonstrating our method creating figure 2 in the paper.  This figure demonstrates the effect of increasing the number of layers we regularise, showing how the saliency map concentrates on the bird as we increase the number of layers. 


## Citation
If you use this code for your project please consider citing us:
```
@inproceedings{Elliott2021PerceptualBall,
  title={Explaining Classifiers using Adversarial Perturbations on the Perceptual Ball},
  author={Elliott, Andrew and Law, Stephen and Russell, Chris},
  booktitle={Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2021}
}
```


## Other references

In addition to acknowledge the other resources in this field, several of which we use in the paper. 

**Visual Explanation Experiments**
* Insertion Deletion [Rise repo](https://github.com/eclique/RISE) 
* Pointing Game [TorchRay](github.com/facebookresearch/TorchRay)

**Alternative Saliency Methods**
For reference, here are links to alternative saliency methods.
We note that if available we use the standard implementation of each methods in each game and use a reference implementation otherwise. see paper for full details. 

* [TorchRay Library - Multiple methods](github.com/facebookresearch/TorchRay)
* [Pytorch Cnn Visualisation - Multiple methods](https://github.com/utkuozbulak/pytorch-cnn-visualizations)
* [NormGrad](www.github.com/ruthcfong/TorchRay/tree/normgrad)
* [RISE](github.com/eclique/RISE)
