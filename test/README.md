
## Notes on reproducing

With the model weights downloaded from Google Drive (see [this issue](https://github.com/alan-turing-institute/ViT-LASNet/issues/2) for a query about where they could be held in a repository with model card to support reuse).

Images are a small test set from [this project on freshwater plankton](https://github.com/NERC-CEH/plankton_ml/).

### ViT model (version = 2, num_classes = 18)

Note this is the version in the subdirectory with `model.safetensors` included, not the single-file `.pt` version.

```
python test.py -w ~/vit_finetuned_MiSLAS_vit_lr5e-05_epochs30/ -o out -n 18 -m 2 -f ../../plankton_ml/tests/fixtures/test_images/
```

### 3 class ResNet18 

With the 3-class Resnet18 weights and the model_version 1 ("combined) we see output:
```
python test.py -w ../ResNet_18_3classes_RGB.pth -o out.csv -m 1 -f ../../plankton_ml/tests/fixtures/test_images/
```

### 18 class ResNet18 

As above but with an `-n` option to specify the model classes (used both to initialise and for predictions). 
Defaults to 3, will raise an error with values other than 18

```
python test.py -w ../ResNet_18_18classes_RGB.pth -o out.csv -m 1 -n 18 -f ../../plankton_ml/tests/fixtures/test_images/
```


